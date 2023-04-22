# Turning Dross Into Gold Loss: Is BERT4Rec really better than SASRec?

Scripts to reproduce main results:

```sh
cd src/

# MovieLens-1m

# SASRec+
python run.py --config-name=SASRec data_path=../data/ml-1m.txt dataset.max_length=200
# SASRec+ 3000
python run.py --config-name=SASRec data_path=../data/ml-1m.txt dataset.max_length=200 +dataset.num_negatives=3000 dataset.full_negative_sampling=False
# SASRec vanilla
python run.py --config-name=SASRec data_path=../data/ml-1m.txt dataset.max_length=200 +seqrec_module.loss=bce +dataset.num_negatives=1
# GRU4Rec
python run.py --config-name=RNN data_path=../data/ml-1m.txt dataset.max_length=200
# BERT4Rec
python run.py --config-name=BERT4Rec data_path=../data/ml-1m.txt dataset.max_length=200

# MovieLens-20m

# SASRec+
python run.py --config-name=SASRec data_path=../data/ml-20m.txt dataset.max_length=200 model_params.hidden_units=256
# SASRec+ 3000
python run.py --config-name=SASRec data_path=../data/ml-20m.txt dataset.max_length=200 +dataset.num_negatives=3000 dataset.full_negative_sampling=False model_params.hidden_units=256
# SASRec vanilla
python run.py --config-name=SASRec data_path=../data/ml-20m.txt dataset.max_length=200 +seqrec_module.loss=bce +dataset.num_negatives=1 model_params.hidden_units=256
# GRU4Rec
python run.py --config-name=RNN data_path=../data/ml-20m.txt dataset.max_length=200 model_params.input_size=256 model_params.hidden_size=256
# BERT4Rec
python run.py --config-name=BERT4Rec data_path=../data/ml-20m.txt dataset.max_length=200 model_params.hidden_size=256

# Beauty

# SASRec+
python run.py --config-name=SASRec data_path=../data/beauty.txt
# SASRec+ 3000
python run.py --config-name=SASRec data_path=../data/beauty.txt +dataset.num_negatives=3000 dataset.full_negative_sampling=False
# SASRec vanilla
python run.py --config-name=SASRec data_path=../data/beauty.txt +seqrec_module.loss=bce +dataset.num_negatives=1
# GRU4Rec
python run.py --config-name=RNN data_path=../data/beauty.txt
# BERT4Rec
python run.py --config-name=BERT4Rec data_path=../data/beauty.txt

# Steam

# SASRec+
python run.py --config-name=SASRec data_path=../data/steam.txt
# SASRec+ 3000
python run.py --config-name=SASRec data_path=../data/steam.txt +dataset.num_negatives=3000 dataset.full_negative_sampling=False
# SASRec vanilla
python run.py --config-name=SASRec data_path=../data/steam.txt +seqrec_module.loss=bce +dataset.num_negatives=1
# GRU4Rec
python run.py --config-name=RNN data_path=../data/steam.txt
# BERT4Rec
python run.py --config-name=BERT4Rec data_path=../data/steam.txt

# Yelp

# SASRec+
python run.py --config-name=SASRec data_path=../data/yelp.txt
# SASRec+ 3000
python run.py --config-name=SASRec data_path=../data/yelp.txt +dataset.num_negatives=3000 dataset.full_negative_sampling=False
# SASRec vanilla
python run.py --config-name=SASRec data_path=../data/yelp.txt +seqrec_module.loss=bce +dataset.num_negatives=1
# GRU4Rec
python run.py --config-name=RNN data_path=../data/yelp.txt
# BERT4Rec
python run.py --config-name=BERT4Rec data_path=../data/yelp.txt
```

BPR-MF code is in a separate notebook `notebooks/BPR-MF.ipynb`.