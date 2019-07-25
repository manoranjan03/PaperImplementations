import pandas as pd



def getResultDataFrame(path):  
    
    try:
        df_losses = pd.read_csv(path)
        
    except:
        df_losses = pd.DataFrame(columns=['epochs',
                                  'train_loss_resnet20',
                                  'val_loss_resnet20',
                                  'train_acc_resnet20',
                                  'val_acc_resnet20',
                                  
                                  'train_loss_resnet32',
                                  'val_loss_resnet32',
                                  'train_acc_resnet32',
                                  'val_acc_resnet32',                                  
                                  
                                  'train_loss_resnet44',
                                  'val_loss_resnet44',
                                  'train_acc_resnet44',
                                  'val_acc_resnet44',
                                  
                                  'train_loss_resnet44',
                                  'val_loss_resnet44',
                                  'train_acc_resnet44',
                                  'val_acc_resnet44',
                                  
                                  'train_loss_resnet56',
                                  'val_loss_resnet56',
                                  'train_acc_resnet56',
                                  'val_acc_resnet56',
                                  
                                  'train_loss_resnet110',
                                  'val_loss_resnet110',
                                  'train_acc_resnet110',
                                  'val_acc_resnet110',
                                  
                                  'train_loss_resnet1202',
                                  'val_loss_resnet1202',
                                  'train_acc_resnet1202',
                                  'val_acc_resnet1202',
                                  
                                  'train_loss_plain_net20',
                                  'val_loss_plain_net20',
                                  'train_acc_plain_net20',
                                  'val_acc_plain_net20',
                                  
                                  'train_loss_plain_net32',
                                  'val_loss_plain_net32',
                                  'train_acc_plain_net32',
                                  'val_acc_plain_net32',
                                  
                                  'train_loss_plain_net44',
                                  'val_loss_plain_net44',
                                  'train_acc_plain_net44',
                                  'val_acc_plain_net44',
                                  
                                  'train_loss_plain_net56',
                                  'val_loss_plain_net56',
                                  'train_acc_plain_net56',
                                  'val_acc_plain_net56',
                                  
                                  'train_loss_plain_net110',
                                  'val_loss_plain_net110',
                                  'train_acc_plain_net110',
                                  'val_acc_plain_net110',
                                  
                                  'train_loss_VGG19',
                                  'val_loss_VGG19',
                                  'train_acc_VGG19',
                                  'val_acc_VGG19'])

        df_losses['epochs'] = np.arange(epochs)
        
        return df_losses           

        