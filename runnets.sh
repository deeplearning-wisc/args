# screen -S train0 -dm bash -c "python3 net.py --batchsize 64 --gpu 0 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "relu" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train1 -dm bash -c "python3 net.py --batchsize 32 --gpu 1 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "relu" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train2 -dm bash -c "python3 net.py --batchsize 64 --gpu 2 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "relu" --testsize 16384 --lr .001 --epochs 120 --printevery 200 --topology 4096 256 128 64 2"

# screen -S train3 -dm bash -c "python3 net.py --batchsize 64 --gpu 3 --seed 10 --experiment "aux2" --optimizer "adam" --activation "relu" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train4 -dm bash -c "python3 net.py --batchsize 32 --gpu 4 --seed 10 --experiment "aux2" --optimizer "adam" --activation "relu" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train5 -dm bash -c "python3 net.py --batchsize 64 --gpu 5 --seed 10 --experiment "aux2" --optimizer "adam" --activation "relu" --testsize 16384 --lr .001 --epochs 120 --printevery 200 --topology 4096 256 128 64 2"

# screen -S train6 -dm bash -c "python3 net.py --batchsize 128 --gpu 6 --seed 10 --experiment "aux2" --optimizer "adam" --activation "linear" --testsize 16384 --lr .001 --epochs 120 --printevery 200 --topology 4096 2"


# wow these should not have been linear activation runs....
# screen -S train7 -dm bash -c "python3 net.py --batchsize 128 --gpu 7 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "linear" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"
# screen -S train8 -dm bash -c "python3 net.py --batchsize 128 --gpu 0 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "linear" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"
# screen -S train9 -dm bash -c "python3 net.py --batchsize 128 --gpu 1 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "linear" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"

# screen -S train10 -dm bash -c "python3 net.py --batchsize 64 --gpu 2 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "linear" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"
# screen -S train11 -dm bash -c "python3 net.py --batchsize 64 --gpu 3 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "linear" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"
# screen -S train12 -dm bash -c "python3 net.py --batchsize 64 --gpu 4 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "linear" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"

# screen -S train13 -dm bash -c "python3 net.py --batchsize 32 --gpu 5 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "linear" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"
# screen -S train14 -dm bash -c "python3 net.py --batchsize 32 --gpu 6 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "linear" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"
# screen -S train15 -dm bash -c "python3 net.py --batchsize 32 --gpu 7 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "linear" --testsize 16384 --lr .01 --epochs 120 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"

# screen -S train16 -dm bash -c "python3 net.py --batchsize 64 --gpu 0 --seed 10 --experiment "aux2" --optimizer "adam" --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"
# screen -S train17 -dm bash -c "python3 net.py --batchsize 64 --gpu 1 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"
# screen -S train18 -dm bash -c "python3 net.py --batchsize 64 --gpu 2 --seed 10 --experiment "aux2" --optimizer "sgd" --activation "relu" --testsize 16384 --lr .01 --epochs 60 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"

# screen -S train19 -dm bash -c "python3 net.py --batchsize 64 --gpu 3 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --standardscaletrain --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"

# screen -S train20 -dm bash -c "python3 net.py --batchsize 64 --gpu 6 --seed 10 --experiment "aux2" --optimizer "sgd" --epochshuffle --standardscaletrain --activation "relu" --testsize 16384 --lr .01 --epochs 60 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"
# screen -S train21 -dm bash -c "python3 net.py --batchsize 64 --gpu 6 --seed 10 --experiment "aux2" --optimizer "sgd" --epochshuffle --activation "relu" --testsize 16384 --lr .01 --epochs 60 --printevery 200 --topology 4096 2048 1024 512 256 128 64 32 16 8 4 2"

# screen -S train22 -dm bash -c "python3 net.py --batchsize 64 --gpu 5 --seed 10 --experiment "aux2" --optimizer "sgd" --epochshuffle --standardscaletrain --activation "relu" --testsize 16384 --lr .01 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train23 -dm bash -c "python3 net.py --batchsize 64 --gpu 7 --seed 10 --experiment "aux2" --optimizer "sgd" --epochshuffle --activation "relu" --testsize 16384 --lr .01 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"

# screen -S train24 -dm bash -c "python3 net.py --batchsize 64 --gpu 0 --seed 10 --experiment "aux2" --optimizer "sgd" --epochshuffle --standardscaletrain  --activation "tanh" --testsize 16384 --lr .01 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train25 -dm bash -c "python3 net.py --batchsize 64 --gpu 2 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --standardscaletrain  --activation "tanh" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"

# screen -S train26 -dm bash -c "python3 net.py --batchsize 256 --gpu 6 --seed 10 --experiment "aux2" --optimizer "sgd" --epochshuffle --activation "relu" --testsize 16384 --lr .01 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train27 -dm bash -c "python3 net.py --batchsize 256 --gpu 6 --seed 10 --experiment "aux2" --optimizer "sgd" --epochshuffle --activation "relu" --testsize 16384 --lr .01 --epochs 60 --printevery 200 --topology 4096 4096 4096 128 2"

# screen -S train28 -dm bash -c "python3 net.py --batchsize 256 --gpu 7 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train29 -dm bash -c "python3 net.py --batchsize 256 --gpu 7 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 4096 4096 128 2"

# screen -S train30 -dm bash -c "python3 net.py --batchsize 512 --gpu 6 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train31 -dm bash -c "python3 net.py --batchsize 1024 --gpu 6 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"

# screen -S train32 -dm bash -c "python3 net.py --batchsize 2048 --gpu 6 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train33 -dm bash -c "python3 net.py --batchsize 4096 --gpu 7 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"

# screen -S train34 -dm bash -c "python3 net.py --batchsize 8192 --gpu 6 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"
# screen -S train35 -dm bash -c "python3 net.py --batchsize 16384 --gpu 7 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 256 128 64 2"

# screen -S train36 -dm bash -c "python3 net.py --batchsize 16384 --gpu 0 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "relu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 2048 512 512 128 64 2"
# screen -S train37 -dm bash -c "python3 net.py --batchsize 16384 --gpu 1 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "silu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 2048 512 512 128 64 2"
# screen -S train38 -dm bash -c "python3 net.py --batchsize 16384 --gpu 0 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "selu" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 2048 512 512 128 64 2"
# screen -S train39 -dm bash -c "python3 net.py --batchsize 16384 --gpu 1 --seed 10 --experiment "aux2" --optimizer "adam" --epochshuffle --activation "tanh" --testsize 16384 --lr .001 --epochs 60 --printevery 200 --topology 4096 2048 512 512 128 64 2"