import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

def plot_loss(model_name):
    data = []

    filename = os.path.join(f'checkpoints/{model_name}/loss_log.txt')

    log_file = open(filename, 'r')
    all_lines = log_file.readlines()

    count = 0

    # Strips the newline character
    for index, line in enumerate(all_lines):

        #drop first line
        if index == 0:
            print('dropping first line of log_file')
        else: 

            # grab last line of each epoch
            if index - 9 == 0 or (index - 9) % 10 == 0:

                info_fields = line.split(' ') 

                tmp = {
                    "epoch": int(info_fields[1].replace(",","")),
                    "iters": int(info_fields[3].replace(",","")),
                    "time": float(info_fields[5].replace(")","")),
                    'GAN': float(info_fields[7]),
                    'GAN_Feat': float(info_fields[9]),
                    'VGG': float(info_fields[11]),
                    'D_Fake': float(info_fields[13]),
                    'D_real': float(info_fields[15])
                }

                data.append(tmp)
                
    df = pd.DataFrame(data)

    plt.rcParams["figure.figsize"] = (16,8)

    plt.plot(df["epoch"], df["GAN"], label='GAN')
    plt.plot(df["epoch"], df["GAN_Feat"], label="GAN_Feat")
    plt.plot(df["epoch"], df["VGG"], label='VGG')

    plt.plot(df["epoch"], df["GAN"]+df["GAN_Feat"]+df["VGG"], label='total generator loss')


    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss for GAN, Gan_Feat, VGG, etc.")
    plt.legend()
    #plt.show()
    plt.savefig(f'checkpoints/{model_name}/loss_log_graph.png')