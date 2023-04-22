# ml-1m
python run.py --config-name=SASRec data_path=../data/ml-1m.txt dataset.max_length=200
python run.py --config-name=RNN data_path=../data/ml-1m.txt dataset.max_length=200
python run.py --config-name=BERT4Rec data_path=../data/ml-1m.txt dataset.max_length=200
python run.py --config-name=SASRec data_path=../data/ml-1m.txt dataset.max_length=200 +seqrec_module.loss=bce +dataset.num_negatives=1
python run.py --config-name=SASRec data_path=../data/ml-1m.txt dataset.max_length=200 +dataset.num_negatives=3000 dataset.full_negative_sampling=False

# beauty
python run.py --config-name=SASRec data_path=../data/beauty.txt
python run.py --config-name=RNN data_path=../data/beauty.txt
python run.py --config-name=BERT4Rec data_path=../data/beauty.txt
python run.py --config-name=SASRec data_path=../data/beauty.txt +seqrec_module.loss=bce +dataset.num_negatives=1
python run.py --config-name=SASRec data_path=../data/beauty.txt +dataset.num_negatives=3000 dataset.full_negative_sampling=False

# steam
python run.py --config-name=SASRec data_path=../data/steam.txt
python run.py --config-name=RNN data_path=../data/steam.txt
python run.py --config-name=BERT4Rec data_path=../data/steam.txt
python run.py --config-name=SASRec data_path=../data/steam.txt +seqrec_module.loss=bce +dataset.num_negatives=1
python run.py --config-name=SASRec data_path=../data/steam.txt +dataset.num_negatives=3000 dataset.full_negative_sampling=False

# yelp
python run.py --config-name=SASRec data_path=../data/yelp.txt
python run.py --config-name=RNN data_path=../data/yelp.txt
python run.py --config-name=BERT4Rec data_path=../data/yelp.txt
python run.py --config-name=SASRec data_path=../data/yelp.txt +seqrec_module.loss=bce +dataset.num_negatives=1
python run.py --config-name=SASRec data_path=../data/yelp.txt +dataset.num_negatives=3000 dataset.full_negative_sampling=False

#ml-20m
python run.py --config-name=SASRec data_path=../data/ml-20m.txt dataset.max_length=200 model_params.hidden_units=256
python run.py --config-name=RNN data_path=../data/ml-20m.txt dataset.max_length=200 model_params.input_size=256 model_params.hidden_size=256
python run.py --config-name=BERT4Rec data_path=../data/ml-20m.txt dataset.max_length=200 model_params.hidden_size=256
python run.py --config-name=SASRec data_path=../data/ml-20m.txt dataset.max_length=200 +seqrec_module.loss=bce +dataset.num_negatives=1 model_params.hidden_units=256
python run.py --config-name=SASRec data_path=../data/ml-20m.txt dataset.max_length=200 +dataset.num_negatives=3000 dataset.full_negative_sampling=False model_params.hidden_units=256