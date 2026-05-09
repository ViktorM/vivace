"""Set group (or tags) on existing wandb runs by name pattern.

Useful when runs were launched before `--wandb-group` existed, or you forgot
to set it, and you want to retroactively cluster runs in the wandb UI's
group view (mean ± stddev across runs in a group).

Requires `wandb login` first (credentials read from ~/.netrc).

Usage:

    # Dry-run first to confirm the filter catches the right runs:
    python -m tests.wandb_regroup \\
        --project viktorm/vivace \\
        --names dapo_rloo_seed42 gspo_rloo_seed42 \\
        --group loss-comparison-qw25-0.5b-500steps \\
        --dry-run

    # Then drop --dry-run to apply:
    python -m tests.wandb_regroup \\
        --project viktorm/vivace \\
        --names dapo_rloo_seed42 gspo_rloo_seed42 \\
        --group loss-comparison-qw25-0.5b-500steps

    # Add tags instead of (or alongside) a group:
    python -m tests.wandb_regroup \\
        --project viktorm/vivace \\
        --names dapo_rloo_compile_seed42 \\
        --add-tags compile_model length_pinned

    # Just list runs without changing anything:
    python -m tests.wandb_regroup --project viktorm/vivace --list
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wandb_regroup")
    p.add_argument("--project", required=True,
                   help='wandb project as "<entity>/<project>", e.g. "viktorm/vivace"')
    p.add_argument("--names", nargs="*", default=None,
                   help="substrings to match against run.name (any match wins). "
                        "Omit to match all runs.")
    p.add_argument("--group", default=None,
                   help="group string to set on matching runs (None = leave unchanged)")
    p.add_argument("--add-tags", nargs="+", default=None,
                   help="tags to add to matching runs (merged with existing tags)")
    p.add_argument("--remove-tags", nargs="+", default=None,
                   help="tags to remove from matching runs")
    p.add_argument("--dry-run", action="store_true",
                   help="list intended changes without modifying anything")
    p.add_argument("--list", action="store_true",
                   help="just print every run's name/id/group/tags and exit")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. `pip install wandb`", file=sys.stderr)
        return 1

    api = wandb.Api()
    runs = list(api.runs(args.project))

    if args.list:
        print(f"{'name':50}  {'id':12}  {'group':40}  tags")
        for r in runs:
            print(f"{r.name:50.50}  {r.id:12}  {(r.group or '(none)'):40.40}  {r.tags}")
        return 0

    if args.group is None and not args.add_tags and not args.remove_tags:
        print("ERROR: nothing to do — pass --group, --add-tags, --remove-tags, or --list",
              file=sys.stderr)
        return 1

    def matches(run) -> bool:
        if args.names is None:
            return True
        return any(n in run.name for n in args.names)

    matched = 0
    for run in runs:
        if not matches(run):
            continue
        matched += 1
        changes = []
        if args.group is not None and run.group != args.group:
            changes.append(f"group: {(run.group or '(none)')} → {args.group}")
        new_tags = list(run.tags)
        if args.add_tags:
            for t in args.add_tags:
                if t not in new_tags:
                    new_tags.append(t)
                    changes.append(f"+tag: {t}")
        if args.remove_tags:
            for t in args.remove_tags:
                if t in new_tags:
                    new_tags.remove(t)
                    changes.append(f"-tag: {t}")
        if not changes:
            print(f"{run.name}  ({run.id})  — already in target state, skipping")
            continue
        prefix = "WOULD " if args.dry_run else ""
        print(f"{prefix}update: {run.name}  ({run.id})")
        for c in changes:
            print(f"    {c}")
        if not args.dry_run:
            if args.group is not None:
                run.group = args.group
            run.tags = new_tags
            run.update()

    print(f"\n{matched} run(s) matched")
    if args.dry_run:
        print("(dry-run — no changes applied; rerun without --dry-run to commit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
