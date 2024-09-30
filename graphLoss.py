import re
import matplotlib.pyplot as plt

#questo po essere sostituito dalla lettura di un file
log_data = """
2024-09-30 16:08:39,844 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 0 Loss: 2607191.5
2024-09-30 16:09:51,501 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 1 Loss: 1528017.0
2024-09-30 16:11:03,411 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 2 Loss: 1077799.5
2024-09-30 16:12:18,157 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 3 Loss: 829637.0
2024-09-30 16:13:32,578 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 4 Loss: 740520.5
2024-09-30 16:14:45,295 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 5 Loss: 655565.0
2024-09-30 16:15:59,520 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 6 Loss: 600685.5
2024-09-30 16:17:14,116 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 7 Loss: 508818.0
2024-09-30 16:18:30,149 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 8 Loss: 444843.0
2024-09-30 16:19:44,982 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 9 Loss: 390256.0
2024-09-30 16:20:59,229 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 10 Loss: 377123.0
2024-09-30 16:22:15,898 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 11 Loss: 358812.0
2024-09-30 16:23:29,197 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 12 Loss: 328800.0
2024-09-30 16:24:44,123 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 13 Loss: 316216.0
2024-09-30 16:25:59,086 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 14 Loss: 263313.0
2024-09-30 16:27:13,776 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 15 Loss: 268345.0
2024-09-30 16:28:29,692 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 16 Loss: 269982.0
2024-09-30 16:29:43,974 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 17 Loss: 240035.0
2024-09-30 16:31:00,225 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 18 Loss: 238494.0
2024-09-30 16:32:18,000 : INFO : Model Word2Vec<vocab=17424091, vector_size=200, alpha=0.1> - Epoch 19 Loss: 223805.0
"""

pattern = r"Epoch (\d+) Loss: ([\d.]+)"
matches = re.findall(pattern, log_data)

epochs = [int(match[0]) for match in matches]
losses = [float(match[1]) for match in matches]

plt.figure(figsize=(10,6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b')

plt.xlabel=('Epoch')
plt.ylabel=('Loss')
plt.title=('Word2Vec Training Loss over Epochs')
plt.grid(True)

#plt.show()
plt.savefig('w2v_loss_plot.png')