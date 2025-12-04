"""
Simple heuristic Clash Royale bot for KataCR.

Uses InteractEnv to:
  - Read state from your phone (cards in hand, elixir, arena / field).
  - Decide a simple action based on that state.
  - Tap on the screen to place cards.

Policy:
  - Every `min_interval` seconds (game time),
      * choose the highest-cost card in hand you can afford,
      * choose a target position based on enemy positions on the arena,
      * play that card.
  - Otherwise, do nothing.

Before running:
  1. Load v4l2 loopback and start scrcpy video sink:
       sudo modprobe v4l2loopback
       scrcpy --v4l2-sink=/dev/video2 --no-video-playback

  2. From the KataCR repo root:
       python -m katacr.policy.heuristic_bot --eval-num 3 --interval 5
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np

# ---------------------------------------------------------------------
# Repo path + imports
# ---------------------------------------------------------------------

# Match the style of eval scripts
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Go 3 levels up from this file: .../KataCR
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))

from katacr.policy.env.interact_env import InteractEnv
from katacr.constants.card_list import card2elixir
from katacr.utils import colorstr


# ---------------------------------------------------------------------
# Simple Heuristic Agent
# ---------------------------------------------------------------------

class SimpleHeuristicAgent:
    """
    Policy that chooses actions based only on:
      - cards in hand (s['cards'])
      - elixir (s['elixir'])
      - arena grid (s['arena'])
      - time (s['time'])

    Action format (same as KataCR):
      [card_slot, x, y, delay]

    Where:
      card_slot in {0,1,2,3,4}
        0 = no-op
        1..4 = hand slots
      x in [0, width-1]
      y in [0, height-1]
      delay is unused here (always 0; we gate by time ourselves).
    """

    def __init__(self, env: InteractEnv, min_interval: float = 5.0):
        self.env = env
        self.idx2card = env.idx2card  # mapping from index -> card name
        self.min_interval = min_interval
        self.last_play_time = 0.0

    # -------- helpers to read state fields --------

    @staticmethod
    def _get_time(s):
        # s['time'] should be a scalar (float/int)
        return float(s.get("time", 0.0))

    @staticmethod
    def _get_elixir(s):
        e = s.get("elixir", None)
        if e is None:
            return 0
        return int(e)

    # -------- choose which card to play --------

    def choose_card_slot(self, s):
        """
        Choose the best hand slot (1..4) to play based on current elixir.
        We pick the highest-cost card we can afford.

        Returns:
          slot in {0,1,2,3,4}  (0 = no playable card)
        """
        elixir = self._get_elixir(s)
        best_slot = 0
        best_cost = -1

        # Hand slots 1..4, consistent with KataCR's policy
        for slot in range(1, 5):
            card_idx = int(s["cards"][slot])
            card_name = self.idx2card.get(str(card_idx), "empty")

            if card_name == "empty":
                continue

            cost = card2elixir.get(card_name, 0)
            if cost <= elixir and cost > best_cost:
                best_cost = cost
                best_slot = slot

        return best_slot

    # -------- choose where to place it on the field --------

    def choose_target_xy(self, s):
        """
        Choose a target (x, y) grid position based on arena.

        Expected arena shape: (H, W, C) ~ (32, 18, channels)
        We treat non-zero in channel 0 as "unit present" and use their
        average position as a target. If no units present, we play at a
        default position on our side of the map.
        """
        arena = s.get("arena", None)

        # Default grid size used in KataCR
        default_h, default_w = 32, 18

        if arena is None:
            x = default_w // 2
            y = int(default_h * 0.75)
            return x, y

        arena = np.array(arena)
        if arena.ndim != 3:
            x = default_w // 2
            y = int(default_h * 0.75)
            return x, y

        H, W, C = arena.shape

        # Use first channel to detect units
        first_channel = arena[..., 0]

        # This threshold may need adjustment depending on encoding
        enemy_mask = first_channel > 0

        if np.any(enemy_mask):
            ys, xs = np.where(enemy_mask)
            x = int(xs.mean())
            y = int(ys.mean())
        else:
            # No visible units -> default to center of our side
            x = W // 2
            y = int(H * 0.75)

        # Clamp to valid range
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        return x, y

    # -------- main policy function --------

    def act(self, s):
        """
        Decide an action for the current state.

        Returns:
          np.array([slot, x, y, delay], dtype=np.int32)
        """
        t = self._get_time(s)

        # Rate-limit: only play every `min_interval` seconds (game time)
        if t - self.last_play_time < self.min_interval:
            return np.array([0, 0, 0, 0], dtype=np.int32)

        # Decide which card to play
        slot = self.choose_card_slot(s)
        if slot == 0:
            # Nothing playable -> no-op
            return np.array([0, 0, 0, 0], dtype=np.int32)

        # Decide where to play (field position)
        x, y = self.choose_target_xy(s)
        self.last_play_time = t

        action = np.array([slot, x, y, 0], dtype=np.int32)
        return action


# ---------------------------------------------------------------------
# Main interaction loop (like Evaluator.eval, but using heuristic agent)
# ---------------------------------------------------------------------

def run_heuristic(eval_num=None, min_interval=5.0, verbose=True):
    """
    Run heuristic episodes using InteractEnv.

    Args:
      eval_num (int|None): number of episodes to run. If None, loops forever.
      min_interval (float): minimum seconds between plays.
      verbose (bool): whether to print per-action logs.
    """
    env = InteractEnv(show=True, save=True)
    agent = SimpleHeuristicAgent(env, min_interval=min_interval)

    episode = 0

    while True:
        episode += 1
        scores = []
        use_actions = 0

        # Reset environment; (s, a0, info0)
        s, a0, info0 = env.reset(auto=eval_num is not None)

        # Initialize last play time from game time
        agent.last_play_time = float(s.get("time", 0.0))
        done = False
        total_reward = 0.0

        print(colorstr(f"Starting episode {episode}"))

        while not done:
            # Decide action based on current state
            a = agent.act(s)

            # Step environment. We ignore delay at env level (max_delay=0)
            s, _, r, done, info = env.step(a, max_delay=0, prob_img=None)

            total_reward += float(r)
            scores.append(total_reward)

            if a[0] != 0:
                use_actions += 1
                if verbose:
                    print(
                        colorstr("Action:"),
                        f"slot={int(a[0])}, xy=({int(a[1])}, {int(a[2])}), "
                        f"time={s['time']:.1f}, elixir={s.get('elixir')}"
                    )

        print(
            colorstr("Episode finished:"),
            f"episode={episode}, total_reward={total_reward:.1f}, "
            f"timesteps={s.get('time', 'N/A')}, use_actions={use_actions}"
        )

        if eval_num is not None and episode >= eval_num:
            break


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simple heuristic KataCR bot")
    parser.add_argument(
        "--eval-num",
        type=int,
        default=3,
        help="Number of episodes to run (None-like for infinite).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Minimum seconds between card plays.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    args = parser.parse_args()

    run_heuristic(
        eval_num=args.eval_num,
        min_interval=args.interval,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()