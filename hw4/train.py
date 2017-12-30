import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import skimage.io
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import imageio
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(123, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        """
        torch.Size([64, 251, 1, 1])
        torch.Size([64, 256, 4, 4])
        torch.Size([64, 512, 8, 8])
        torch.Size([64, 128, 16, 16])
        torch.Size([64, 64, 32, 32])
        torch.Size([64, 3, 64, 64])
        """
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(x)), 0.2)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.tanh(self.deconv5(x))/2.0+0.5
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512+23, 512, 4, 1, 0)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512,1)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label): # input 3*64*64; label 151*1*1
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        y = label.expand(64,23,x.size()[-1],x.size()[-1])
        x = torch.cat([x, y], 1) # channel 
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = x.view([64,-1])
        x = self.linear(x)
        x = F.sigmoid(x)

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d)  or (m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# read test text for texting 
with open("sample_testing_text.txt", "r") as f:
    test_text = [line.strip().split(",")[1] for line in f.readlines()]
def get_color(text):
    print(text)
    e = False
    h = False
    text = text.split(" ")
    for i,t in enumerate(text):
        if t == "hair":
            hair_color = text[i-1] + " hair"
            h = True
        if t == "eyes":
            eyes_color = text[i-1] + " eyes"
            e = True
    if h == False:
        hair_color = "blonde hair"
    if e == False:
        eyes_color = "blue eyes"
    return hair_color, eyes_color
for i in test_text:
    print(get_color(i))



#create specific color 
with open("hair_encoder.pkl","rb") as f:
    hair_encoder = pickle.load(f)
with open("eyes_encoder.pkl","rb") as f:
    eyes_encoder = pickle.load(f)
# label preprocess (onehot embedding matrix)
hair_onehot = torch.zeros(12, 12)
hair_onehot = hair_onehot.scatter_(1, torch.LongTensor(list(range(12))).view(12,1), 1).view(12, 12, 1, 1)

eyes_onehot = torch.zeros(11, 11)
eyes_onehot = eyes_onehot.scatter_(1, torch.LongTensor(list(range(11))).view(11,1), 1).view(11, 11, 1, 1)

print("hair_onehot",hair_onehot.size())
print("eyes_onehot",eyes_onehot.size())
# special mission ("red hair", "green eyes")
print(hair_encoder["red hair"])
print(eyes_encoder["green eyes"])




# create testing tensor
tensor = []
for i in test_text:
    hair_color, eyes_color = get_color(i)
    y_special = torch.cat([hair_onehot[hair_encoder[hair_color]].view(1,12,1,1),
                           eyes_onehot[eyes_encoder[eyes_color]].view(1,11,1,1)],1)
    tensor.append(y_special.repeat(10,1,1,1))
    print(y_special.size())
y_special = torch.cat(tensor)


z_special = torch.randn((100, 100)).view(-1, 100, 1, 1)
fixed_z_, fixed_y_label_ = Variable(z_special.cuda(), volatile=True), Variable(y_special.cuda(), volatile=True)





def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    G.eval()
    test_images = G(fixed_z_, fixed_y_label_)
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(10, 10))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0)))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = True, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()



# training parameters
batch_size = 64
lr = 0.0002
train_epoch = 300

# data_loader
img_size = 64
with open("combine_img_v4.pkl","rb") as f:
    image_X = pickle.load(f)
#     image_X = torch.stack(image_X)
hair_list_index = torch.from_numpy(np.load("combine_hair_v4.npy"))
eyes_list_index = torch.from_numpy(np.load("combine_eyes_v4.npy"))


perm_index = torch.randperm(len(image_X))
image_X = image_X[perm_index]
hair_list_index = hair_list_index[perm_index]
eyes_list_index = eyes_list_index[perm_index]
# image_X, encoded_text = shuffle(image_X, encoded_text)
# image_X = image_X.transpose(0,3,1,2)
print("train image shape", image_X.shape)
print("train encode shape", eyes_list_index.shape)
# network
G = generator(64)
D = discriminator(64)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
root = 'anime_cDCGAN_results/'
model = 'anime_cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


# In[17]:


print('training start!')
start_time = time.time()
updates = 0
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    
    # shuffle
    perm_index = torch.randperm(len(image_X))
    image_X = image_X[perm_index]
    hair_list_index = hair_list_index[perm_index]
    eyes_list_index = eyes_list_index[perm_index]
    
    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 2
        D_optimizer.param_groups[0]['lr'] /= 2
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 2
        D_optimizer.param_groups[0]['lr'] /= 2
        print("learning rate change!")

    epoch_start_time = time.time()
    y_real_ = torch.ones(batch_size) #batch size = 128
    y_fake_ = torch.zeros(batch_size) 
    y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
    for batch_index in list(range(0,len(image_X),batch_size)):
        if batch_index + batch_size > len(image_X):
            break
        updates+=1
        mini_batch = batch_size
        # train discriminator D
        for _ in range(1):
            x_ = image_X[batch_index:batch_index+batch_size]
            y_true_h = hair_list_index[batch_index:batch_index+batch_size]
            y_true_e = eyes_list_index[batch_index:batch_index+batch_size]
            D.zero_grad()

            #(real img, right text)

            y_label_ = torch.cat([hair_onehot[y_true_h],eyes_onehot[y_true_e]],1) #[batch size,23,1,1]
            x_, y_label_ = Variable(x_.cuda()), Variable(y_label_.cuda()) # x_ : [128, 1, 64, 64]
            D_result_1 = D(x_, y_label_).squeeze() #(real img, right text): 1
            D_real_loss = BCE_loss(D_result_1, y_real_) # real picture & real label

            # (fake img, right text)
            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1) # z_ : [(128, 100, 1, 1)]
            y_label_ = torch.cat([hair_onehot[y_true_h],eyes_onehot[y_true_e]],1) 
            z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())
            G_result = G(z_, y_label_) # generate fake img
            D_result_2 = D(G_result, y_label_).squeeze() #(fake img, right text): 0
            D_fake_loss_1 = BCE_loss(D_result_2, y_fake_)

            #(real img, wrong text)
            y_false_h = (torch.rand(mini_batch, 1) * 12).type(torch.LongTensor).squeeze().clamp(0,11) # random sample y (text)
            y_false_e = (torch.rand(mini_batch, 1) * 11).type(torch.LongTensor).squeeze().clamp(0,10)
            y_label_ = torch.cat([hair_onehot[y_false_h],eyes_onehot[y_false_e]],1)
            y_label_ = Variable(y_label_.cuda())
            D_result_3 = D(x_, y_label_).squeeze()
            D_fake_loss_2 = BCE_loss(D_result_3, y_fake_)

            # (wrong img, right text )
            perm_index = torch.randperm(len(x_))
            x_wrong = x_.cpu().data[perm_index]
            y_label_ = torch.cat([hair_onehot[y_true_h],eyes_onehot[y_true_e]],1)
            x_wrong, y_label_ = Variable(x_wrong.cuda()), Variable(y_label_.cuda()) # x_ : [128, 1, 64, 64]
            D_result_3 = D(x_wrong, y_label_).squeeze() #(real img, right text): 1
            D_fake_loss_3 = BCE_loss(D_result_3, y_fake_) # wrong picture & real label

            D_train_loss = D_real_loss + D_fake_loss_1 + D_fake_loss_2 + D_fake_loss_3
    #         D_train_loss = D_real_loss + D_fake_loss_1 + D_fake_loss_2 

            D_train_loss.backward()
            D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        # train generator G (fake img, real text)
        for _ in range(5):
            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            y_label_ = torch.cat([hair_onehot[y_true_h],eyes_onehot[y_true_e]],1)
            z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())

            G_result = G(z_, y_label_)
            D_result = D(G_result, y_label_).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

        G_losses.append(G_train_loss.data[0])
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, 
                                                                 torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    if epoch %20 == 0:
        torch.save(G.state_dict(), root + model + 'generator_param_'+str(epoch)+'.pkl')
        torch.save(D.state_dict(), root + model + 'discriminator_param_'+str(epoch)+'.pkl')
        with open(root + model + 'train_hist.pkl', 'wb') as f:
            pickle.dump(train_hist, f)
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), root + model + 'generator_param.pkl')
torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(skimage.io.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)