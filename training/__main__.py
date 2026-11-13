"""Task-first training entry; e.g. python -m training --task push_t --model act ..."""
import argparse
from importlib import import_module
from tasks.stages import stage_adapter

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--task', required=True)
    parser.add_argument('--model', choices=('act','diffusion'), required=True)
    args, remaining=parser.parse_known_args(argv)
    stage_adapter(args.task, 'training').validate_model(args.model)
    return import_module('training.'+args.model+'.train').main(['--task',args.task,*remaining])

if __name__ == '__main__':
    main()
