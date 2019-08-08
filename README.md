# virtual env

### create a new env
```
mkvirtualenv deeplearningWorkshop
```

### use an existing env
```
workon deeplearningWorkshop
```

### exit env
```
deactivate
```


# Install requirements
```
pip3 install -r requirements
```


# Test Data
At first you will need to verify your kaggle account and accept the competition terms and conditions. Follow this link to do so (https://www.kaggle.com/c/dogs-vs-cats/rules).

Then you can use resources/prepare_data.sh script to configure the directories and download the training and validation data
```
./resources/prepare_data.sh
```


# Visualisation using tensorboard
```
tensorboard --logdir tensorboard
```

