import matplotlib.pyplot as plt
import os

def init_utils2():
    os.remove("data/mean.dat")
    os.remove("data/rew.dat")

def save_mean_value(value):
    f = open("data/mean.dat","a")
    f.write("%f\n" % value)
    f.close()
def save_rew(value):
    f = open("data/rew.dat","a")
    f.write("%f\n" % value)
    f.close()

def save_mean_value2(v_value, n):
    f = open("data/mean" + str(n) + ".dat","a")
    for i in v_value:
        f.write("%f\n" % i)
    f.close()
    
def save_rew2(v_value, n):
    f = open("data/rew" + str(n) + ".dat","a")
    for i in v_value:
        f.write("%f\n" % i)
    f.close()

def make_plot():
    plt.clf()
    # Fill the vector mean with the data from mean.dat
    f = open("mean.dat","r")
    mean = []
    i = 0
    line = f.readline()
    while line:
        i += 1
        if (i % 1000 == 0):
            print(i)
        mean.append(round(float(line), 2))
        line = f.readline()
    f.close()

    # Fill the vector rewards with the file from rew.dat
    f = open("rew.dat","r")
    rew = []
    line = f.readline()
    while line:
        i += 1
        if (i % 1000 == 0):
            print(i)
        rew.append(round(float(line), 2))
        line = f.readline()
    f.close()
    
    plt.plot(mean)
    plt.plot(rew)
    plt.legend(["Reward", "average"])
    plt.title("Reward history")
    plt.savefig("./plot.png")