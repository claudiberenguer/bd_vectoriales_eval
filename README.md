# INSTALACIÓN DEL REPOSITORIO

0. clonar el repositorio
1. copiar una GEMINI_AP_KEY valida el archivo llamado .env (está en la raíz)
2. en la raíz, ejecutar `uv venv --python 3.12 .venv`
3. en la raíz ejecutar `source .venv/bin/activate` (linux); si en windows, .venv\Scripts\activate
4. en la raíz, ejectuar `uv pip install -r requirements.txt`
5. en la raíz ejecutar `uv run --active --with jupyter jupyter lab`, con lo que dará unos link que al clicar van a abrir jupyter lab en el navegador por defecto
6. se usa el propio notebook como documento de entrega. es `notebooks_ii/ENTREGA.ipynb`
7. abrir `notebooks_ii/ENTREGA.ipynb` y seleccinar arriba ala derecha el kernel, asignado el entorno virtual creado en paso 2 (.venv)
8. Ejecutar de forma secuencial el notebook. En él está toda la información

Contenido del repositorio:
  - `archivo .env` donde copiar una GEMINI_API_KEY válida
  - `directorio notebooks_ii`. Contiene:
      - ENTREGA.ipynb: contiene un índice que aproximadamente mapea los puntos del ejercico. Tabién contiene una sección con los docstring de utils.py para consulta rápida
      - `utils.py`: este archivo contiene funciones que se usan en ENTREGA.ipynb. Prácticamente toda la lógica está en ese archivo
  - `irectorio datos`. contiene los datos del ejercicio
  - `directorio db`. Es donde se crea la base de datos ChromaDB desde, desde ENTREGA.ipynb
  - `artifacts`. algunos archivos de resultados
      
