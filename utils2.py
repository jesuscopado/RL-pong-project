def save_mean_value(value):
    f = open("mean.dat","a")
    f.write("%f\n" % value)
    f.close()
def save_rew(value):
    f = open("rew.dat","a")
    f.write("%f\n" % value)
    f.close()
