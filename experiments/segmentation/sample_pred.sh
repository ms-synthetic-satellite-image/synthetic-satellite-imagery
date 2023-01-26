#!/bin/bash
# predict on real image
# inputfolder =  "/home/group-segmentation/main/segmentation/results/experiments/"
# outputfolder = "./results/experiments/res/"
i=0
for file in /home/group-segmentation/main/segmentation/results/experiments/real_image/*
do 
    python test.py --model checkpoints/syn_1e-5_p10_epoch5/checkpoint_epoch5.pth --label_nc 6 --input "$file" --output ./results/experiments/res/"real_${i}.png"
    python test.py --model checkpoints/syn_1e-5_p5_epoch5_1000_2/checkpoint_epoch5.pth --label_nc 6 --input "$file" --output ./results/experiments/res/"syn1000_${i}.png"
    python test.py --model checkpoints/syn_1e-5_p5_epoch5_2000_2/checkpoint_epoch5.pth --label_nc 6 --input "$file" --output ./results/experiments/res/"syn2000_${i}.png"
    i+=1
    echo $i
done

i=0
for file in /home/group-segmentation/main/segmentation/results/experiments/synthesized_image/*
do 
    python test.py --model checkpoints/syn_1e-5_p10_epoch5/checkpoint_epoch5.pth --label_nc 6 --input "$file" --output ./results/experiments/res/"real_syn_${i}.png"
    python test.py --model checkpoints/syn_1e-5_p5_epoch5_1000_2/checkpoint_epoch5.pth --label_nc 6 --input "$file" --output ./results/experiments/res/"syn1000_syn_${i}.png"
    python test.py --model checkpoints/syn_1e-5_p5_epoch5_2000_2/checkpoint_epoch5.pth --label_nc 6 --input "$file" --output ./results/experiments/res/"syn2000_syn_${i}.png"
    i+=1
    echo $i
done

# python test.py --model checkpoints/syn_1e-5_p10_epoch5/checkpoint_epoch5.pth --label_nc 6 --input /home/group-segmentation/main/segmentation/results/experiments/real_image/m_3807608_nw_18_1_naip-new_cp5.tif --output ./results/experiments/res/real.png

# python test.py --model checkpoints/syn_1e-5_p5_epoch5_1000_2/checkpoint_epoch5.pth --label_nc 6 --input /home/group-segmentation/main/segmentation/results/experiments/real_image/m_3807608_nw_18_1_naip-new_cp5.tif --output ./results/experiments/res/syn1000.png

# python test.py --model checkpoints/syn_1e-5_p5_epoch5_2000_2/checkpoint_epoch5.pth --label_nc 6 --input /home/group-segmentation/main/segmentation/results/experiments/real_image/m_3807608_nw_18_1_naip-new_cp5.tif --output ./results/experiments/res/syn2000.png

# python test.py --model checkpoints/syn_1e-5_p10_epoch5/checkpoint_epoch5.pth --label_nc 6 --input /home/group-segmentation/main/segmentation/results/experiments/synthesized_image/m_3807608_nw_18_1_naip-new_cp5.tif --output ./results/experiments/res/real_syn.png

# python test.py --model checkpoints/syn_1e-5_p5_epoch5_1000_2/checkpoint_epoch5.pth --label_nc 6 --input /home/group-segmentation/main/segmentation/results/experiments/synthesized_image/m_3807608_nw_18_1_naip-new_cp5.tif --output ./results/experiments/res/syn1000_syn.png

# python test.py --model checkpoints/syn_1e-5_p5_epoch5_2000_2/checkpoint_epoch5.pth --label_nc 6 --input /home/group-segmentation/main/segmentation/results/experiments/synthesized_image/m_3807608_nw_18_1_naip-new_cp5.tif --output ./results/experiments/res/syn2000_syn.png


