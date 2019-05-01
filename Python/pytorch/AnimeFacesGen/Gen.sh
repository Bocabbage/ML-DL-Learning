OUTPIC_PATH=./
OUTPARA_PATH=./
DATASET_PATH=./
PARA_G=./parag.pth
PARA_D=./parad.pth

NGPU=1
EPOCH=1

#conda activate py37
python ./DCGAN.py --opipath $OUTPIC_PATH --opapath $OUTPARA_PATH --dpath $DATASET_PATH\
--ngpu NGPU --epoch EPOCH  > ./train_log