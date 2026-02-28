## Driver para la operación matriz - matriz (GEMM)

El siguiente driver tiene como objetivo la evaluación del alumnado para comprobar los conocimientos adquiridos a lo que respecta la optimización del producto matriz - matriz. 
Como el objeto de estudio es la optimización bajo plataformas NEON ARM, el código está preparado para ser ejecutado bajo este tipo de plataformas.

## Funcionamiento

En primer lugar hay que compilar el driver mediante el comando: make

Una vez compilado, el siguiente paso es fijar desde el fichero "run_driver.sh" el tipo de ejecución. En primer lugar se determina el tipo de Test sobre el que se va a evaluar el rendimiento, 
y en segundo lugar se fija la dificultad. Cada nivel de dificultad corresponde con una técnica diferente de optimización, el objetivo del alumnado es desarrollar su propia rutina que lleve 
a cabo una GEMM capaz de batir el rendimiento objetivo. 


