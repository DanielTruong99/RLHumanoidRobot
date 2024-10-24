## Overview

- `Jumping over obstacle policy` with heigh scan
- Continuous pd
- Walking policy running at 100 hz

## Train

```bash
python scripts/rsl_rl/train.py --task LegRobot-leap-v1 --num_envs 4096 --headless
```

## Play with Trained Policy

Follow these step:

1. **Add folder logs** to enable the search model path:

   - Create a `logs/leg_leap_6/2024-10_24_02-41-37`
   - Place the model located at `.data/jump_over_obstacle_position_cmds.pt` to the created folder

2. **Run play script**:
   - Run the play scipt
   ```bash
       python scripts/rsl_rl/play.py --task LegRobot-leap-play-v1 --num_envs 1
   ```
