# Predicción de población por cada país para los próximos 4 años
#### 2021 07 12, FE Charry-Pastrana, charrypastranaernesto@gmail.com

Procedimiento seguido: 
1. Exploración de datos

    1.1 **Visualización** de los datos: comportamiento exponencial.
    
    1.2. **Correlación de** Población total con otras Variables (Net migration, Mortality rate - adult - male and female, Fertility rate). 
    
    
2. Modelado de datos: 

    2.1 Función **exponencial** para todo el intervalo temporal. 
    
    2.2. Función exponencial para los últimos 20 años.
    
    2.3. Función exponencial para los últimos 20 años + **Método de Monte Carlo** básico. 
    
    2.4. Función exponencial a trozos (**ventana de 4 años**) **+** corrección (**promedio** de dos años anteriores)
    
    2.5. Método lineal incluyendo **otras variables**, P(t, x_i) = Date + Variable_i
    
    2.6. Método lineal incluyendo otras variables y utilizando Monte Carlo básico para predecir Problación.


Teoría utilizada: 
- Comportamiento líneal.
- Correlación. 
- Probabilidad (Monte Carlo).

***
⌨️ with ❤️! 📌
