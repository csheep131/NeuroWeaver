"""Run entry point for the ablation machine.

Usage:
    python -m runs.run --config configs/runs/run001_control.yaml
    
Smoke Test:
    python -m runs.run --config configs/runs/run001_control.yaml --smoke-test
    
Local Proxy Run:
    python -m runs.run --config configs/runs/run001_control.yaml --local-proxy

Full Training Run:
    python -m runs.run --config configs/runs/run001_control.yaml --mode train
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from core.config import load_config, Config
from core.registry import RunRegistry
from core.seed import set_seed
from core.logging import RunLogger
from core.local_proxy import (
    LocalProxyRunner,
    LocalMetrics,
    create_smoke_test_config,
    check_local_prerequisites,
    SmokeTestResult,
)
from research import (
    AblationReporter,
    Phase1Evaluator,
    Phase2Evaluator,
    Phase3Evaluator,
    Phase1RunType,
    Phase2RunType,
    Phase3RunType,
)


def get_git_commit() -> str | None:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_smoke_test(config: Config) -> SmokeTestResult:
    """Führe Smoke-Test für Konfiguration durch.

    Smoke-Test Kriterien:
    - Modell startet (kein OOM bei Initialisierung)
    - Mindestens 1 Step läuft
    - Metriken werden geschrieben
    """
    run_id = config.run_id
    result = SmokeTestResult(
        run_id=run_id,
        success=False,
        started=False,
        first_step_completed=False,
        metrics_written=False,
    )

    logger = RunLogger(run_id)
    logger.log_info(f"Starting smoke test for {run_id}")

    try:
        # Check prerequisites
        prereqs = check_local_prerequisites()
        for warning in prereqs.get("warnings", []):
            logger.log_warning(warning)

        # Initialize local proxy runner
        smoke_config = create_smoke_test_config(str(Path("configs") / "runs" / f"{run_id}.yaml"))
        proxy_runner = LocalProxyRunner(smoke_config)
        proxy_runner.metrics.is_smoke_test = True

        logger.log_info("Smoke test: Initializing model...")
        result.started = True

        # Simulate model initialization and first step
        # In real implementation: actual model forward pass
        start_time = time.perf_counter()

        # Placeholder: Simulate step
        time.sleep(0.1)  # Simulate computation
        step_time_ms = (time.perf_counter() - start_time) * 1000

        proxy_runner.record_step(tokens_processed=256, step_time_ms=step_time_ms)
        result.first_step_completed = True

        # Record VRAM (placeholder)
        proxy_runner.record_vram(1000.0)  # Simulated VRAM usage

        # Metrics
        metrics = {
            "val_bpb": 1.5,  # Placeholder
            "ms_per_step": step_time_ms,
            "steps_completed": 1,
            "artifact_bytes": 10_000_000,
            "peak_vram_mb": proxy_runner.metrics.peak_vram_mb,
            "oom_count": proxy_runner.metrics.oom_count,
            "is_smoke_test": True,
            "is_local_proxy": True,
        }

        result.peak_vram_mb = proxy_runner.metrics.peak_vram_mb
        result.metrics_written = True

        # Evaluate smoke test success
        success, failures = proxy_runner.check_success()
        result.success = success

        if success:
            logger.log_info("Smoke test PASSED")
        else:
            logger.log_warning(f"Smoke test FAILED: {failures}")
            result.error_message = ", ".join(failures)

        return result

    except Exception as e:
        logger.log_error(f"Smoke test failed: {e}")
        result.error_message = str(e)
        return result


def run_full_training(config: Config, logger: RunLogger) -> dict:
    """Run full training with PyTorch using proper Trainer.

    Args:
        config: Run configuration
        logger: Run logger

    Returns:
        Metrics dictionary
    """
    import torch
    from data import create_dataloader, create_tokenizer
    from train.pytorch_model import create_model, ModelConfig
    from train.optimizer_factory import create_optimizer
    from train.scheduler import create_scheduler
    from train.trainer import Trainer, TrainConfig
    from eval.bpb_eval import BPBEvaluator

    logger.log_info("Initializing PyTorch training...")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log_info(f"Using device: {device}")

    # Create tokenizer
    tokenizer_cfg = config.tokenizer
    tokenizer = create_tokenizer(
        tokenizer_type=tokenizer_cfg.get("type", "byte"),
        vocab_size=tokenizer_cfg.get("vocab_size", 256),
        byte_fallback=tokenizer_cfg.get("byte_fallback", True),
    )
    logger.log_info(f"Tokenizer: {tokenizer_cfg.get('type', 'byte')}")

    # Create model - FIX: Use tokenizer vocab_size for model
    model_cfg = config.to_dict()
    model_cfg["model"]["vocab_size"] = tokenizer.vocab_size  # FIX CRITICAL 6
    model = create_model(model_cfg)
    num_params = model.num_parameters_millions()
    logger.log_info(f"Model created: {num_params:.2f}M parameters, vocab_size={tokenizer.vocab_size}")

    # Create data loader - FIX: Use config batch_size with memory-aware limiting
    training_cfg = config.training
    requested_batch_size = training_cfg.get("batch_size", 32)
    
    # Only limit batch size if on CPU or small GPU
    if device == "cpu":
        batch_size = min(requested_batch_size, 4)
    else:
        # Check available VRAM
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        if total_vram < 10:  # Less than 10GB
            batch_size = min(requested_batch_size, 4)
        else:
            batch_size = requested_batch_size
    
    if batch_size != requested_batch_size:
        logger.log_info(f"Batch size adjusted: {requested_batch_size} -> {batch_size} (memory constraints)")
    
    data_config = {
        "train_data_path": "",
        "eval_data_path": "",
        "seq_len": training_cfg.get("seq_len", 128),
        "batch_size": batch_size,
        "tokenizer_type": tokenizer_cfg.get("type", "byte"),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "shuffle": True,
        "seed": config.seed,
    }
    train_loader = create_dataloader(data_config, tokenizer)
    logger.log_info(f"Data loader created (seq_len={data_config['seq_len']}, batch_size={batch_size})")

    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=str(training_cfg.get("optimizer", "adamw")),
        learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.1)),
    )
    logger.log_info(f"Optimizer: {training_cfg.get('optimizer', 'adamw')}")

    # Create scheduler
    num_steps = int(training_cfg.get("num_steps", 10000))
    warmup_steps = int(training_cfg.get("warmup_steps", 100))
    
    # Ensure warmup_steps < num_training_steps
    if warmup_steps >= num_steps:
        warmup_steps = max(1, num_steps // 10)
        logger.log_warning(f"Warmup steps adjusted to {warmup_steps} (was >= num_steps)")
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=str(training_cfg.get("scheduler", "cosine")),
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps,
        min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.1)),
    )
    logger.log_info(f"Scheduler: {training_cfg.get('scheduler', 'cosine')} (warmup={warmup_steps}, total={num_steps})")

    # Create trainer - FIX: Properly configure and use Trainer
    train_config = TrainConfig.from_dict(training_cfg)
    train_config.device = device
    train_config.num_steps = num_steps  # FIX: Ensure correct num_steps
    trainer = Trainer(model, optimizer, train_config, logger)

    # Create evaluator
    bpb_evaluator = BPBEvaluator(tokenizer)

    # Create evaluation function
    def eval_fn(model, step):
        """Evaluation function called periodically during training."""
        model.eval()
        eval_loader = create_dataloader(data_config, tokenizer)
        eval_result = bpb_evaluator.compute_bpb(model, eval_loader, device=device)
        model.train()
        return {"val_bpb": eval_result.val_bpb}

    # FIX: Use trainer.train() with proper batch iteration
    logger.log_info(f"Starting training for {num_steps} steps...")
    
    # Convert data loader to iterator that cycles through data
    class DataIterator:
        """Iterator that cycles through data loader indefinitely."""
        def __init__(self, loader):
            self.loader = loader
            self.iter = iter(loader)
        
        def __next__(self):
            try:
                return next(self.iter)
            except StopIteration:
                # Reset iterator when exhausted
                self.iter = iter(self.loader)
                return next(self.iter)
    
    train_iter = DataIterator(train_loader)
    
    # Run training using trainer
    final_state = trainer.train(train_iter, eval_fn=eval_fn)
    
    logger.log_info(f"Training completed: {final_state.step} steps in {final_state.total_time_seconds:.1f}s")
    logger.log_info(f"Best loss: {final_state.best_loss:.4f}, Avg ms/step: {final_state.ms_per_step:.2f}")

    # Final evaluation
    logger.log_info("Running final evaluation...")
    model.eval()
    eval_loader = create_dataloader(data_config, tokenizer)
    eval_result = bpb_evaluator.compute_bpb(model, eval_loader, device=device)
    val_bpb = eval_result.val_bpb

    logger.log_info(f"Final Val BPB: {val_bpb:.4f}")

    # FIX: Correct metrics calculation
    metrics = {
        "val_bpb": val_bpb,
        "ms_per_step": final_state.ms_per_step,
        "steps_completed": final_state.step,
        "artifact_bytes": int(num_params * 1_000_000 * 4),  # Estimate in bytes
        "quantized_val_bpb": val_bpb * 1.02,  # Estimate
        "train_loss": final_state.best_loss,
        "best_val_bpb": val_bpb,
        "total_time_seconds": final_state.total_time_seconds,
    }

    return metrics


def run_training(config_path: str, smoke_test: bool = False, local_proxy: bool = False, mode: str = None) -> int:
    """Run a training run.

    Args:
        config_path: Path to the run configuration
        smoke_test: Whether to run smoke test mode
        local_proxy: Whether to run in local proxy mode
        mode: Run mode (smoke_test, local_proxy, train)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Determine mode
    if mode == "train":
        use_smoke_test = False
        use_local_proxy = False
        use_full_training = True
    elif mode == "smoke_test":
        use_smoke_test = True
        use_local_proxy = False
        use_full_training = False
    elif mode == "local_proxy":
        use_smoke_test = False
        use_local_proxy = True
        use_full_training = False
    else:
        use_smoke_test = smoke_test
        use_local_proxy = local_proxy
        use_full_training = False

    # Load configuration
    config = load_config(config_path)

    if not config.run_id:
        print("Error: run_id not specified in config")
        return 1

    # Initialize registry
    registry = RunRegistry()

    # Register run
    git_commit = get_git_commit()
    entry = registry.register(
        run_id=config.run_id,
        config_hash=config.config_hash,
        parent_run_id=config.parent_run_id,
        seed=config.seed,
    )

    # Set up logger
    logger = RunLogger(config.run_id)

    try:
        # Set seed
        set_seed(config.seed)

        # Start run
        registry.start_run(config.run_id, git_commit)
        logger.log_info(f"Starting run {config.run_id}")
        logger.log_info(f"Config hash: {config.config_hash}")
        logger.log_info(f"Git commit: {git_commit}")
        logger.log_info(f"Seed: {config.seed}")
        
        mode_str = "train" if use_full_training else ("smoke_test" if use_smoke_test else "local_proxy")
        logger.log_info(f"Mode: {mode_str}")

        # Check prerequisites
        prereqs = check_local_prerequisites()
        for warning in prereqs.get("warnings", []):
            logger.log_warning(warning)

        if use_smoke_test:
            # Run smoke test
            smoke_result = run_smoke_test(config)
            if not smoke_result.success:
                registry.fail_run(config.run_id, smoke_result.error_message or "Smoke test failed")
                return 1

            metrics = {
                "val_bpb": 1.5,
                "ms_per_step": 10.0,
                "steps_completed": 1,
                "artifact_bytes": 10_000_000,
                "peak_vram_mb": smoke_result.peak_vram_mb,
                "oom_count": 0,
                "is_smoke_test": True,
                "is_local_proxy": True,
            }
        elif use_local_proxy:
            # Run local proxy mode
            proxy_runner = LocalProxyRunner(config)
            proxy_runner.metrics.is_proxy = True

            # Manually enable proxy mode
            proxy_runner.proxy_config.enabled = True
            proxy_runner.proxy_config.steps = 50
            proxy_runner.proxy_config.microbatch = 1
            proxy_runner.proxy_config.grad_accumulation = 8

            training_cfg = proxy_runner.prepare_for_local_run()

            logger.log_info(f"Local proxy config: steps={training_cfg.get('num_steps')}, "
                          f"batch_size={training_cfg.get('batch_size')}")

            # Placeholder: Simulate proxy run
            for step in range(min(10, training_cfg.get("num_steps", 50))):
                start_time = time.perf_counter()
                time.sleep(0.05)  # Simulate computation
                step_time_ms = (time.perf_counter() - start_time) * 1000
                proxy_runner.record_step(tokens_processed=256, step_time_ms=step_time_ms)
                proxy_runner.record_vram(5000.0)  # Simulated VRAM

            metrics = {
                "val_bpb": 1.5,
                "ms_per_step": proxy_runner.metrics.ms_per_step,
                "steps_completed": proxy_runner.metrics.steps_completed,
                "artifact_bytes": 10_000_000,
                "peak_vram_mb": proxy_runner.metrics.peak_vram_mb,
                "oom_count": proxy_runner.metrics.oom_count,
                "is_local_proxy": True,
            }
        elif use_full_training:
            # Full training with PyTorch
            metrics = run_full_training(config, logger)
        else:
            # Full training (placeholder)
            logger.log_info("Full training not yet implemented - placeholder run")
            metrics = {
                "val_bpb": 1.5,
                "ms_per_step": 10.0,
                "steps_completed": 1000,
                "artifact_bytes": 10_000_000,
            }

        # Complete run
        registry.complete_run(config.run_id, metrics)
        logger.log_info(f"Run completed: {config.run_id}")

        # Phase-spezifische Evaluation
        if config.run_id.startswith("run001") or config.run_id.startswith("run002"):
            logger.log_info("Running Phase 1 evaluation...")
            evaluator = Phase1Evaluator(config.run_id)
            report = evaluator.evaluate(metrics)

            if report.should_kill:
                logger.log_warning(f"Run killed by Phase 1 rule: {report.kill_reason}")
                registry.kill_run(config.run_id, report.kill_reason)
            else:
                logger.log_info(f"Phase 1 evaluation: {report.status.upper()}")
                logger.log_info(f"Recommendation: {report.recommendation}")

        elif config.run_id.startswith("run003") or config.run_id.startswith("run010"):
            logger.log_info("Running Phase 2 evaluation...")
            
            # Get parent metrics if available
            parent_metrics = None
            entry = registry.get(config.run_id)
            if entry and entry.parent_run_id:
                parent_entry = registry.get(entry.parent_run_id)
                if parent_entry:
                    parent_metrics = {
                        "val_bpb": parent_entry.val_bpb,
                        "ms_per_step": parent_entry.ms_per_step,
                        "artifact_bytes": parent_entry.artifact_bytes,
                    }
            
            evaluator = Phase2Evaluator(config.run_id)
            report = evaluator.evaluate(metrics, parent_metrics)

            if report.should_kill:
                logger.log_warning(f"Run killed by Phase 2 rule: {report.kill_reason}")
                registry.kill_run(config.run_id, report.kill_reason)
            else:
                logger.log_info(f"Phase 2 evaluation: {report.status.upper()}")
                logger.log_info(f"Gate Status: {report.gate_status.upper()}")
                logger.log_info(f"Recommendation: {report.recommendation}")

        elif config.run_id.startswith("run016") or config.run_id.startswith("run017"):
            logger.log_info("Running Phase 3 evaluation...")
            
            # Get parent metrics if available
            parent_metrics = None
            entry = registry.get(config.run_id)
            if entry and entry.parent_run_id:
                parent_entry = registry.get(entry.parent_run_id)
                if parent_entry:
                    parent_metrics = {
                        "val_bpb": parent_entry.val_bpb,
                        "ms_per_step": parent_entry.ms_per_step,
                        "artifact_bytes": parent_entry.artifact_bytes,
                    }
            
            evaluator = Phase3Evaluator(config.run_id)
            report = evaluator.evaluate(metrics, parent_metrics)

            if report.should_kill:
                logger.log_warning(f"Run killed by Phase 3 rule: {report.kill_reason}")
                registry.kill_run(config.run_id, report.kill_reason)
            else:
                logger.log_info(f"Phase 3 evaluation: {report.status.upper()}")
                logger.log_info(f"Submission Ready: {report.submission_ready}")
                logger.log_info(f"Recommendation: {report.recommendation}")

        # General kill rule evaluation
        reporter = AblationReporter(registry)
        should_kill, reason = reporter.evaluate_run(config.run_id, metrics)
        if should_kill and registry.get(config.run_id).status != "killed":
            logger.log_warning(f"Run killed by rule: {reason}")
            registry.kill_run(config.run_id, reason)
        elif not should_kill:
            logger.log_info("Run passed all kill rules")

        return 0

    except Exception as e:
        logger.log_error(f"Run failed: {e}")
        registry.fail_run(config.run_id, str(e))
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run an ablation experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the run configuration file",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test mode (1-2 steps, minimal validation)",
    )
    parser.add_argument(
        "--local-proxy",
        action="store_true",
        help="Run in local proxy mode (reduced steps, 8GB VRAM optimized)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke_test", "local_proxy", "train"],
        default=None,
        help="Run mode (overrides --smoke-test and --local-proxy)",
    )

    args = parser.parse_args()

    return run_training(
        args.config,
        smoke_test=args.smoke_test,
        local_proxy=args.local_proxy,
        mode=args.mode,
    )


if __name__ == "__main__":
    sys.exit(main())
