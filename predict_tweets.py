"""Batch three-way social sentiment using the saved phase-two model."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from phase2 import predict, ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=ROOT/'results/phase2/model.joblib')
    parser.add_argument('--output',type=Path,default=ROOT/'results/phase2/local/new_predictions.csv')
    args=parser.parse_args()
    frame=pd.read_csv(args.data)
    result=predict(frame,joblib.load(args.model)['model'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False,float_format='%.8f')
    print(f'Scored {len(result)} rows')

if __name__=='__main__':
    main()
