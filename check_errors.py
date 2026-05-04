import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import traceback
import sys

notebook_filename = 'ml_pipeline.ipynb'

try:
    with open(notebook_filename, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600)
    
    print("Execution en cours pour verifier les erreurs...")
    ep.preprocess(nb, {'metadata': {'path': './'}})
    print("Notebook execute avec SUCCES sans aucune erreur !")
except Exception as e:
    print("ERREUR TROUVEE PENDANT L'EXECUTION :\\n")
    print(str(e))
    # We can also extract the exact cell and traceback
    sys.exit(1)
