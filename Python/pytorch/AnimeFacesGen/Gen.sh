OUTPIC_PATH=/home/ljw/AnimeFacesGen/out_pic
OUTPARA_PATH=/home/ljw/AnimeFacesGen/out_para
# DATASET_PATH=/home/ljw/AnimeFacesGen/AnimeFaces
# PARA_G=./parag.pth
# PARA_D=./parad.pth

NGPU=1
EPOCH=1

#conda activate py37
python ./DCGAN.py --opipath $OUTPIC_PATH --opapath $OUTPARA_PATH > ./train_log