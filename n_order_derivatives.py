import data_loader as dtl
import matplotlib.pyplot as plt

def discrete_first_derivative(x):
    dx = [x[1] - x[0]]
    for i in range(1, len(x) - 1):
        dx.append((x[i + 1] - x[i - 1]) / 2)
    dx.append(x[-1] - x[-2])
    return dx

data = dtl.get_data()
total_confirmed = data['totalconfirmed']
daily_confirmed = data['dailyconfirmed']
x = total_confirmed
for i in range(5):
    dx = discrete_first_derivative(x)
    plt.plot(x)
    plt.plot(dx)
    plt.plot(daily_confirmed)
    plt.legend(['d'+str(i), 'd'+str(i + 1)])
    plt.savefig(str(i) + '.png')
    plt.clf()
    x = dx