## Overview

- stable walking with minimal sensor
- Continuous pd
- Walking policy running at 100 hz

## Train

walking

```bash
python scripts/rsl_rl_normal/train.py --task LegRobot-planar-walk-v2 --num_envs 4096 --headless
```

running

```bash
python scripts/rsl_rl_normal/train.py --task LegRobot-planar-walk-v2 --num_envs 8192 --headless
```

## Play with Trained Policy

Follow these step:

1. **Add folder logs** to enable the search model path:

   - Create a `logs/simple_walking_robot/2024-10_24_02-41-37`
   - Place the model located at `.data/walking_policy/model_9999.pt` or `.data/running_policy/model_4850.pt` to the created folder

2. **Run play script**:
   - Run the play scipt
   ```bash
       python scripts/rsl_rl_normal/play.py --task LegRobot-planar-walk-play-v2 --num_envs 1
   ```
