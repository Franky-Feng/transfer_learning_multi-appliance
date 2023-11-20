import matplotlib.pyplot as plt
import numpy as np

appliances = ["Kettle", "Microwave", "Fridge", "Dishwasher", "Washing machine"]
power_consumption1 = [61.6254417, 25, 43.59557867, 40.01379786, 32.96560675]
power_consumption2 = [0, 56.62520281, 41.17017481, 59.84023239, 48.55604503]
# power_consumption1 = [61.75373134, 34.90364026, 70.0610998, 17.94453507, 15.130674]
# power_consumption2 = [0, 15.61643836, 3.510758777, 5.359477124, 6.33059789]
plt.figure(figsize=(8, 4))

x = np.arange(len(appliances))

plt.bar(x - 0.2, power_consumption1, width=0.4, label='UK-DALE', color='#336699', edgecolor='black')
plt.bar(x + 0.2, power_consumption2, width=0.4, label='REDD', color='#FF9900', edgecolor='black')

plt.ylabel('Imp', fontname='Times New Roman', fontsize=16)
plt.title('MAE', fontname='Times New Roman', fontsize=20)

plt.xticks(x, appliances, rotation=0, fontname='Times New Roman', fontsize=14)
plt.yticks(fontname='Times New Roman', fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
plt.legend()
plt.savefig('mae_plot.jpg', format='jpg', bbox_inches='tight')
plt.show()
