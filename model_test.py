from ConvModel import ConvModel

obj = ConvModel(elu=True,img_channel=1,classes=2)
obj.test(2,1,device='cpu')