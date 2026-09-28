from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.run import main
if __name__ == '__main__':
    main(default_method='remem_i')
