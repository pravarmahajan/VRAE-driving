#THEANO_FLAGS="optimizer=None,exception_verbosity=high,floatX=float32" python run.py --traj_data data/smallSample_5_20 --rnn_size 16 --latent_size 32 --batch_size 8 --scale 40 --num_epochs 10
#THEANO_FLAGS="floatX=float32" python run.py --traj_data data/smallSample_5_20 --rnn_size 16 --latent_size 32 --batch_size 8 --scale 40 --num_epochs 10
THEANO_FLAGS="floatX=float32" python run.py --num_drivers 50 --num_trajs 200  --rnn_size 64 --latent_size 64 --batch_size 8 --scale 10 --num_epochs 10 --lamda1 0.0 --lamda2 1.0 --suffix _Hd
