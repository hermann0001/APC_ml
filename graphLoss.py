import re
import matplotlib.pyplot as plt

#questo po essere sostituito dalla lettura di un file
log_data = {0: 340422.65625, 1: 377510.96875, 2: 287529.0, 3: 353973.625, 4: 268551.0, 5: 313233.625, 6: 313266.125, 7: 313737.5, 8: 358772.5, 9: 223503.25, 10: 268882.5, 11: 313804.0, 12: 312664.5, 13: 257311.75, 14: 149792.0, 15: 179334.0, 16: 179167.0, 17: 180908.0, 18: 215047.5, 19: 221173.5, 20: 230089.0, 21: 201553.5, 22: 199855.5, 23: 192256.5, 24: 186792.5, 25: 177923.0, 26: 173520.0, 27: 194141.5, 28: 161940.0, 29: 156631.0, 30: 177710.5, 31: 145815.0, 32: 114367.5, 33: 151585.5, 34: 126909.0, 35: 120519.5, 36: 117556.0, 37: 131566.5, 38: 88144.0, 39: 114604.0, 40: 70604.0, 41: 84599.0, 42: 98098.0, 43: 97328.0, 44: 69375.0, 45: 81038.0, 46: 92686.0, 47: 79316.0, 48: 78982.0, 49: 78061.0, 50: 78155.0, 51: 75273.0, 52: 99437.0, 53: 87088.0, 54: 87713.0, 55: 84182.0, 56: 72941.0, 57: 71512.0, 58: 83699.0, 59: 70780.0, 60: 81690.0, 61: 81505.0, 62: 70604.0, 63: 68642.0, 64: 79051.0, 65: 80393.0, 66: 68689.0, 67: 67596.0, 68: 68198.0, 69: 66045.0, 70: 55731.0, 71: 76842.0, 72: 55284.0, 73: 75605.0, 74: 76503.0, 75: 65145.0, 76: 63456.0, 77: 62706.0, 78: 73754.0, 79: 62530.0, 80: 69880.0, 81: 71010.0, 82: 62112.0, 83: 72660.0, 84: 79783.0, 85: 68560.0, 86: 69915.0, 87: 68902.0, 88: 50085.0, 89: 49913.0, 90: 71587.0, 91: 61032.0, 92: 71747.0, 93: 81677.0, 94: 68735.0, 95: 76666.0, 96: 58400.0, 97: 60220.0, 98: 60229.0, 99: 69127.0}

#pattern = r"(\d+): ([\d.,]+)"
#matches = re.findall(pattern, log_data)

epochs = list(log_data.keys())
losses = list(log_data.values())

# Creazione del grafico
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, losses, marker='o', linestyle='-', color='b')

# Aggiungi etichette e titolo
fig.suptitle(r'Loss function per $\alpha = 0.1$', fontsize=14, fontweight='bold') # Titolo con la lettera greca alpha
ax.grid(True)

# Imposta i tick sull'asse X ogni 10 epoche
ax.set_xticks(range(0, 101, 10))  # Mostra solo i numeri da 0 a 100 con step di 10

# Imposta le etichette degli assi
ax.set_xlabel('EPOCHS')  # Etichetta asse X
ax.set_ylabel('LOSS')    # Etichetta asse Y
#plt.show()
plt.savefig('figures/w2v_loss_plot2.png')