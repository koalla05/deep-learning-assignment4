"""Exercise a complete two-stage run with a tiny, entirely local encoder."""

import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from ner.artifacts import latest_checkpoint, publish_bundle, verify_bundle, write_json
from ner.cv import export_fold_ensemble
from ner.data import DataSpec
from ner.experiment import ensemble_directory, record_kaggle_result
from ner.inference import predict_saved
from ner.predict import average_probs
from ner.preflight import memory_check
from ner.runtime import load_initial_model, resolved_precision
from ner.train import (
    GradFlowCallback,
    RecoveryHistoryCallback,
    RunConfig,
    WeightedTokenTrainer,
    load_probs,
    run,
    run_or_load,
)
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast,
    TrainingArguments,
)


class ArtifactIntegrityTests(unittest.TestCase):
    def test_smoke_ensembles_cannot_replace_production_exports(self):
        full = ensemble_directory(Path("drive/runs"), "xlmr-recipe")
        smoke = ensemble_directory(Path("drive/runs_smoke"), "xlmr-recipe")
        self.assertNotEqual(full, smoke)
        self.assertEqual(smoke.parts[-2], "ensembles_smoke")

    def test_t4_uses_fp16_instead_of_emulated_bf16(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=False) as supported,
        ):
            self.assertEqual(resolved_precision("auto"), "fp16")
        supported.assert_called_once_with(including_emulation=False)

    def test_partial_checkpoint_does_not_hide_an_older_recoverable_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            older = Path(temporary) / "checkpoint-250"
            write_json(older / "trainer_state.json", {"global_step": 250})
            publish_bundle(older, [older / "trainer_state.json"])
            partial = Path(temporary) / "checkpoint-500"
            partial.mkdir()
            self.assertEqual(latest_checkpoint(temporary), str(older))

    def test_accumulation_keeps_the_same_gradient_for_posts_of_different_lengths(self):
        class TokenHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(2, 3)

            def forward(self, features):
                return SimpleNamespace(logits=self.head(features))

        model = TokenHead()
        features = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[2.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]
        )
        labels = torch.tensor([[0, 1, -100], [2, -100, -100]])
        weights = torch.tensor([1.0, 2.0])
        settings = SimpleNamespace(label_smoothing=0.1)
        WeightedTokenTrainer.compute_loss(
            settings,
            model,
            {
                "features": features,
                "labels": labels,
                "weight": weights,
            },
        ).backward()
        expected = model.head.weight.grad.clone()
        model.zero_grad()
        for row in range(2):
            WeightedTokenTrainer.compute_loss(
                settings,
                model,
                {
                    "features": features[row : row + 1],
                    "labels": labels[row : row + 1],
                    "weight": weights[row : row + 1],
                },
                num_items_in_batch=3,
            ).backward()
        torch.testing.assert_close(model.head.weight.grad, expected)

    def test_a_metrics_file_alone_does_not_make_a_run_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "unfinished"
            directory.mkdir()
            (directory / "metrics.json").write_text("{}")
            cfg = RunConfig(name="unfinished")
            with patch("ner.train.run", return_value={"restarted": True}) as restart:
                self.assertEqual(run_or_load(cfg, temporary), {"restarted": True})
            restart.assert_called_once_with(cfg, temporary, None)

    def test_a_damaged_file_invalidates_the_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = directory / "metrics.json"
            write_json(result, {"f1": 0.8})
            publish_bundle(directory, [result])
            verify_bundle(directory)
            result.write_text("{}")
            with self.assertRaisesRegex(ValueError, "damaged artifact metrics.json"):
                verify_bundle(directory)

    def test_ensemble_members_cannot_silently_drop_a_post(self):
        post = (np.array([[0, 4]]), np.array([[0.1, 0.8, 0.1]]))
        with self.assertRaisesRegex(ValueError, "same"):
            average_probs([[post], [post, post]])


class LocalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "initial-model"
        self.model_dir.mkdir()
        vocabulary = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "у",
            "києві",
            "дощ",
            "львові",
        ]
        vocab_file = self.model_dir / "vocab.txt"
        vocab_file.write_text("\n".join(vocabulary), encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
        tokenizer.save_pretrained(self.model_dir)
        model = BertForTokenClassification(
            BertConfig(
                vocab_size=len(vocabulary),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=24,
                max_position_embeddings=32,
                num_labels=3,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                id2label={0: "O", 1: "B-LOC", 2: "I-LOC"},
                label2id={"O": 0, "B-LOC": 1, "I-LOC": 2},
            )
        )
        model.save_pretrained(self.model_dir)
        self.frame = pd.DataFrame(
            {
                "text": ["У Києві дощ", "У Львові дощ"],
                "spans": [[[2, 7]], [[2, 8]]],
                "weight": [1.0, 1.0],
            }
        )
        self.sets = {
            name: ([self.frame.text.iloc[i]], [[place]])
            for name, i, place in [
                ("cv", 0, "Києві"),
                ("synthetic", 1, "Львові"),
                ("half_A", 0, "Києві"),
                ("half_B", 1, "Львові"),
                ("gold26", 0, "Києві"),
            ]
        }
        self.cfg = RunConfig(
            name="two-stage",
            model=str(self.model_dir),
            data=DataSpec(synthetic="none"),
            stage2_data=DataSpec(sentences=0),
            epochs=1,
            stage2_epochs=1,
            batch_size=1,
            eval_batch_size=1,
            inference_batch_size=1,
            grad_accumulation=2,
            max_length=16,
            stride=4,
            precision="fp32",
            gradient_checkpointing=True,
            checkpoint_steps=1,
            grad_flow_every=1,
        )
        dependencies = {
            "ner.train.provenance": {"inputs": {}, "source": {}, "packages": {}},
            "ner.train.training_frame": self.frame,
            "ner.train.eval_sets": self.sets,
            "ner.cv.eval_sets": self.sets,
            "ner.data.validation_frames": {"sentences": self.frame},
            "ner.train.load_test_posts": pd.DataFrame(
                {
                    "text_id": [7, 8],
                    "half": ["A", "B"],
                    "text": list(self.frame.text),
                }
            ),
        }
        dependencies["ner.cv.load_test_posts"] = dependencies["ner.train.load_test_posts"]
        for target, result in dependencies.items():
            patcher = patch(target, return_value=result)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_fp16_checkpoint_loads_float32_weights_for_training(self):
        # mDeBERTa ships half-precision weights; AMP needs float32 parameters to unscale gradients.
        checkpoint = BertForTokenClassification.from_pretrained(self.model_dir).half()
        checkpoint.save_pretrained(self.model_dir)

        model, tokenizer = load_initial_model(self.cfg)
        inputs = tokenizer("У Києві дощ", return_tensors="pt")
        model(**inputs, labels=torch.zeros_like(inputs["input_ids"])).loss.backward()

        self.assertEqual({p.dtype for p in model.parameters()}, {torch.float32})
        self.assertEqual(
            {p.grad.dtype for p in model.parameters() if p.grad is not None},
            {torch.float32},
        )

    def test_amp_overflow_does_not_break_checkpoint_history(self):
        model = BertForTokenClassification.from_pretrained(self.model_dir)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        # Clipping an overflowing AMP gradient can leave NaNs before the scaler skips the update.
        model.classifier.weight.grad.fill_(float("nan"))
        gradients = GradFlowCallback(every=1)
        state = SimpleNamespace(global_step=0)
        gradients.on_pre_optimizer_step(None, state, None, model=model)

        checkpoint = self.root / "checkpoint-1"
        checkpoint.mkdir()
        state.global_step = 1
        callback = RecoveryHistoryCallback(SimpleNamespace(history=[]), gradients, 0, 0)
        callback.on_save(SimpleNamespace(output_dir=str(self.root)), state, None)

        verify_bundle(checkpoint)
        observed = json.loads((checkpoint / "observations.json").read_text())["grad_flow"][0]
        self.assertFalse(observed["finite"])
        self.assertIsNone(observed["head"])
        self.assertGreater(observed["embeddings"], 0)

    def test_amp_overflow_norm_can_be_saved_in_trainer_logs(self):
        model = BertForTokenClassification.from_pretrained(self.model_dir)
        trainer = WeightedTokenTrainer(
            model=model,
            args=TrainingArguments(output_dir=str(self.root / "trainer"), report_to="none"),
        )
        trainer.log({"loss": 0.5, "grad_norm": float("nan")})
        write_json(self.root / "history.json", trainer.state.log_history)

        observed = json.loads((self.root / "history.json").read_text())[0]
        self.assertIsNone(observed["grad_norm"])
        self.assertEqual(observed["loss"], 0.5)

    def test_notebooks_complete_the_learning_rate_and_three_fold_workflow(self):
        import matplotlib.pyplot as plt

        from ner import recipes

        notebook_dir = Path(__file__).resolve().parents[1] / "transformers"
        optimization = {
            "epochs": 1,
            "batch_size": 2,
            "max_length": 16,
            "stride": 4,
            "precision": "fp32",
        }
        stage2 = {"stage2_data": self.cfg.stage2_data, "stage2_epochs": 1, "stage2_lr": 2e-5}

        # Execute the actual notebook cells against the tiny encoder and fixture rows. Full-sized
        # model downloads, GPU capacity and external authentication are separate Colab checks.
        with (
            patch.object(recipes, "OPTIMIZATION", optimization),
            patch.object(recipes, "STAGE2", stage2),
            patch("ner.experiment.freeze_model_revision", return_value=None),
            patch(
                "ner.preflight.memory_check",
                side_effect=lambda cfg, output: memory_check(cfg, output, require_cuda=False),
            ),
        ):
            for path in sorted(notebook_dir.glob("0[345]*.ipynb")):
                with self.subTest(notebook=path.name):
                    context = {"OUT": self.root / path.stem / "runs", "WANDB_PROJECT": None}
                    cells = json.loads(path.read_text())["cells"]
                    # Setup is checked with a real Drive mount and W&B login on Colab.
                    for cell in cells[3:]:
                        if cell["cell_type"] != "code":
                            continue
                        source = re.sub(
                            r"^MODEL = .+$",
                            f"MODEL = {str(self.model_dir)!r}",
                            "".join(cell["source"]),
                            flags=re.MULTILINE,
                        )
                        exec(compile(source, str(path), "exec"), context)

                    self.assertEqual(len(context["SELECTION"]["trials"]), 2)
                    self.assertEqual(len(set(context["FOLD_RUNS"])), 3)
                    verify_bundle(context["ENSEMBLE_DIR"])
                    self.assertEqual(len(context["restored_predictions"]), 2)
                    plt.close("all")

    def test_two_stages_export_weights_that_reproduce_saved_predictions(self):
        metrics = run(self.cfg, self.root)
        directory = self.root / self.cfg.name
        verify_bundle(directory)
        self.assertIn("half_B", metrics)
        self.assertFalse((directory / "work").exists())
        saved = load_probs(self.root, self.cfg.name)
        _, restored = predict_saved(directory, self.sets["half_A"][0], precision="fp32")
        np.testing.assert_array_equal(restored[0][0], saved["probs"]["half_A"][0][0])
        np.testing.assert_allclose(restored[0][1], saved["probs"]["half_A"][0][1], atol=3e-4)
        self.assertEqual(saved["rows"]["half_A"][0]["text_id"], 7)
        with patch("ner.train.run", side_effect=AssertionError("finished run retrained")):
            self.assertEqual(run_or_load(self.cfg, self.root), metrics)

    def test_ensemble_and_kaggle_record_preserve_the_submission_identity(self):
        run(self.cfg, self.root)
        directory = self.root / "ensemble"
        metrics, submission = export_fold_ensemble(
            self.root,
            [self.cfg.name],
            directory,
            {"selected_lr": self.cfg.learning_rate},
            pd.DataFrame({"half_B": [0.5]}, index=[self.cfg.name]),
        )
        verify_bundle(directory)
        self.assertEqual(list(submission.text_id), [7, 8])
        self.assertIn("gold26", metrics)
        record_kaggle_result(directory, 0.75, "12345")
        result = json.loads((directory / "kaggle/12345-public.json").read_text())
        self.assertEqual(result["score"], 0.75)
        self.assertEqual(len(result["submission_sha256"]), 64)
        # Adding a dated Kaggle score does not modify the frozen inference bundle.
        verify_bundle(directory)
        with self.assertRaises(FileExistsError):
            record_kaggle_result(directory, 0.8, "12345")
        # Rerunning the notebook reuses the frozen ensemble; it cannot replace its submission.
        repeated, _ = export_fold_ensemble(
            self.root,
            [self.cfg.name],
            directory,
            {"selected_lr": self.cfg.learning_rate},
            pd.DataFrame({"half_B": [0.5]}, index=[self.cfg.name]),
        )
        self.assertEqual(repeated, metrics)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            export_fold_ensemble(
                self.root,
                [self.cfg.name],
                directory,
                {"selected_lr": 0.1},
                pd.DataFrame({"half_B": [0.5]}, index=[self.cfg.name]),
            )
        offline_export = self.root / "ensemble-wandb-offline"
        with (
            patch("ner.experiment.log_ensemble", side_effect=ConnectionError),
            self.assertWarns(UserWarning),
        ):
            export_fold_ensemble(
                self.root,
                [self.cfg.name],
                offline_export,
                {},
                pd.DataFrame(),
                wandb_project="offline-test",
            )
        verify_bundle(offline_export)
        self.assertEqual(
            json.loads((offline_export / "logging.json").read_text())["status"],
            "upload_failed",
        )

    def test_memory_preflight_does_not_change_the_initial_weights(self):
        from ner.artifacts import sha256

        initial = sha256(self.model_dir / "model.safetensors")
        report = memory_check(self.cfg, self.root / "memory.json", require_cuda=False)
        self.assertGreater(report["parameters"], 0)
        self.assertEqual(report["precision"], "fp32")
        self.assertEqual(sha256(self.model_dir / "model.safetensors"), initial)

    def test_finished_first_stage_is_restored_after_a_disconnect(self):
        from ner.train import RecoveryHistoryCallback

        cfg = replace(self.cfg, stage2_epochs=2)
        varied = pd.concat([self.frame] * 4, ignore_index=True)
        varied["text"] = [text + " дощ" * i for i, text in enumerate(varied.text)]
        varied["weight"] = [1.0 + i / 10 for i in range(len(varied))]
        source = patch("ner.train.training_frame", return_value=varied)
        source.start()
        self.addCleanup(source.stop)
        run(replace(cfg, name="uninterrupted"), self.root)
        real_train = WeightedTokenTrainer.train
        real_save = RecoveryHistoryCallback.on_save
        interrupted = False

        def disconnect_after_checkpoint(callback, args, state, control, **kwargs):
            nonlocal interrupted
            real_save(callback, args, state, control, **kwargs)
            if "stage-2" in args.output_dir and state.global_step == 5 and not interrupted:
                interrupted = True
                raise RuntimeError("Colab disconnected")

        with patch.object(RecoveryHistoryCallback, "on_save", disconnect_after_checkpoint):
            with self.assertRaisesRegex(RuntimeError, "Colab disconnected"):
                run(cfg, self.root)
        self.assertTrue((self.root / cfg.name / "work/stage-1/complete.json").exists())
        self.assertFalse((self.root / cfg.name / "complete.json").exists())
        resumed_stages = []

        def resume(trainer, *args, **kwargs):
            resumed_stages.append((trainer.args.output_dir, kwargs.get("resume_from_checkpoint")))
            return real_train(trainer, *args, **kwargs)

        with patch.object(WeightedTokenTrainer, "train", resume):
            run(cfg, self.root)
        self.assertEqual(len(resumed_stages), 1)
        self.assertIn("stage-2", resumed_stages[0][0])
        self.assertIn("checkpoint-5", resumed_stages[0][1])
        verify_bundle(self.root / cfg.name)
        restored = BertForTokenClassification.from_pretrained(self.root / cfg.name / "model")
        reference = BertForTokenClassification.from_pretrained(self.root / "uninterrupted/model")
        for name, parameter in restored.state_dict().items():
            np.testing.assert_array_equal(
                parameter.numpy(), reference.state_dict()[name].numpy(), err_msg=name
            )
        history = json.loads((self.root / cfg.name / "history.json").read_text())
        self.assertEqual({row["stage"] for row in history["post_eval"]}, {1, 2})


if __name__ == "__main__":
    unittest.main()
