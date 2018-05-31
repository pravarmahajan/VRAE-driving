#THEANO_FLAGS="optimizer=None,exception_verbosity=high,floatX=float32" python run.py --traj_data data/smallSample_5_20 --rnn_size 16 --latent_size 32 --batch_size 8 --scale 40 --num_epochs 10
THEANO_FLAGS="floatX=float32" python run.py --traj_data data/smallSample_5_20 --rnn_size 16 --latent_size 32 --batch_size 8 --scale 40 --num_epochs 10
