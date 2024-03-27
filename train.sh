export CUDA_VISIBLE_DEVICES=0

steps=200000
name='basesac'
alpha=0.1
max=3
for i in `seq 2 $max`
do
    echo "Iteration $i"
    name='basesac'
    python Lunar_dqn.py \
        --name ${name} \
        --total-steps ${steps} \
        --alpha ${alpha}


    name='revsac'

    python Lunar_dqn.py \
        --name ${name} \
        --total-steps ${steps} \
        --alpha ${alpha}

    name='tianbasesac'

    python Lunar_dqn.py \
        --name ${name} \
        --total-steps ${steps} \
        --alpha ${alpha}

   
done