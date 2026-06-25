import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ustawienie stylu wykresu
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


# Definicja wypukłej funkcji kosztu (Log-Loss w uproszczeniu to taka 'misa')
def cost_function(x, y):
    return x ** 2 + 0.5 * y ** 2


# Generowanie siatki przestrzeni wag
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = cost_function(X, Y)

# Rysowanie przezroczystej powłoki 3D (tzw. "siodła" / misy)
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Symulacja algorytmu Spadku Gradientu
learning_rate = 0.1
iterations = 15
path_x, path_y, path_z = [], [], []

# Punkt startowy (losowe, nieoptymalne wagi początkowe)
current_x, current_y = -8, -8

for _ in range(iterations):
    path_x.append(current_x)
    path_y.append(current_y)
    path_z.append(cost_function(current_x, current_y))

    # Obliczanie gradientów: dJ/dx = 2x, dJ/dy = y
    grad_x = 2 * current_x
    grad_y = current_y

    # Aktualizacja wag (krok w dół gradientu)
    current_x = current_x - learning_rate * grad_x
    current_y = current_y - learning_rate * grad_y

# Dodanie ostatniego punktu
path_x.append(current_x)
path_y.append(current_y)
path_z.append(cost_function(current_x, current_y))

# Rysowanie kuleczki i ścieżki Spadku Gradientu
ax.plot(path_x, path_y, path_z, color='red', marker='o', markersize=6, linewidth=2, label='Ścieżka optymalizatora')
ax.scatter([0], [0], [0], color='black', s=80, label='Minimum globalne (optymalne wagi)', zorder=5)

# Opisy osi
ax.set_xlabel('Waga $\\beta_1$')
ax.set_ylabel('Waga $\\beta_2$')
ax.set_zlabel('Koszt modelu $Cost(\\beta)$')
ax.view_init(elev=35, azim=-55)  # Ustawienie dobrego kąta kamery
ax.legend()

# Zapis do pliku wektorowego (najlepsza jakość do LaTeXa)
plt.savefig('gradient_descent.pdf', bbox_inches='tight', transparent=True)
print("Wykres został wygenerowany i zapisany jako 'gradient_descent.pdf'")