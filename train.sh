echo "Starting training 1... \n"
python atari_ddqn.py --train --env PongNoFrameskip-v4

sleep 120

echo "Starting training 2... \n"
python atari_ddqn.py --train --env PongNoFrameskip-v4

sleep 120

echo "Starting training 3... \n"
python atari_ddqn.py --train --env PongNoFrameskip-v4

sleep 120

echo "Starting training 4... \n"
python atari_ddqn.py --train --env PongNoFrameskip-v4

sleep 120

echo "Starting training 5... \n"
python atari_ddqn.py --train --env PongNoFrameskip-v4
