import os
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim
from GRU_cell import GRUCell
import numpy as np
from torch.distributions import Normal, kl_divergence
import datetime
import dateutil.tz
import argparse

def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()

def concat(*data_list):
    return torch.cat(data_list, 1)

class Sprites(torch.utils.data.Dataset):
    def __init__(self, path, size):
        self.path = path
        self.length = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.load(self.path + '/%d.sprite' % (idx + 1))


class FullQDisentangledVAE(nn.Module):
    def __init__(self, frames, z_dim, conv_dim, hidden_dim, device):
        super(FullQDisentangledVAE, self).__init__()
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.block_size = 3
        self.device = device

        self.z_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.z_mean_prior = nn.Linear(self.z_dim, self.z_dim)
        self.z_logvar_prior = nn.Linear(self.z_dim, self.z_dim)

        self.z_to_c_fwd_list = [GRUCell(input_size=self.z_dim, hidden_size=self.z_dim//self.block_size).to(device)
                                for i in range(self.block_size)]

        self.z_w_function = nn.Linear(self.z_dim, self.block_size)

        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.drop2 = nn.Dropout2d(0.4)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop3 = nn.Dropout2d(0.4)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout2d(0.4)
        self.conv_fc = nn.Linear(4 * 4 * 256, self.conv_dim)  # 4*4 is size 256 is channels
        self.drop_fc = nn.Dropout(0.4)
        self.bnf = nn.BatchNorm1d(self.conv_dim)

        self.deconv_fc = nn.Linear(self.z_dim, 4 * 4 * 256)  # 4*4 is size 256 is channels
        self.deconv_bnf = nn.BatchNorm1d(4 * 4 * 256)
        self.drop_fc_deconv = nn.Dropout(0.4)
        self.deconv4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.dbn4 = nn.BatchNorm2d(256)
        self.drop4_deconv = nn.Dropout2d(0.4)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.dbn3 = nn.BatchNorm2d(256)
        self.drop3_deconv = nn.Dropout2d(0.4)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.dbn2 = nn.BatchNorm2d(256)
        self.drop2_deconv = nn.Dropout2d(0.4)
        self.deconv1 = nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        nonlinearity='relu')  # Change nonlinearity to 'leaky_relu' if you switch
        nn.init.xavier_normal_(self.deconv1.weight, nn.init.calculate_gain('tanh'))

    def encode_frames(self, x):
        x = x.view(-1, 3, 64, 64)  # Batchwise stack the 8 images for applying convolutions parallely
        x = F.leaky_relu(self.conv1(x), 0.1)  # Remove batchnorm, the encoder must learn the data distribution
        x = self.drop2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        x = self.drop3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))
        x = self.drop4(F.leaky_relu(self.bn4(self.conv4(x)), 0.1))
        x = x.view(-1, 4 * 4 * 256)  # 4*4 is size 256 is channels
        x = self.drop_fc(F.leaky_relu(self.bnf(self.conv_fc(x)), 0.1))
        x = x.view(-1, self.frames, self.conv_dim)
        return x

    def decode_frames(self, zf):
        x = zf.view(-1, self.z_dim)  # For batchnorm1D to work, the frames should be stacked batchwise
        x = self.drop_fc_deconv(F.leaky_relu(self.deconv_bnf(self.deconv_fc(x)), 0.1))
        x = x.view(-1, 256, 4, 4)  # The 8 frames are stacked batchwise
        x = self.drop4_deconv(F.leaky_relu(self.dbn4(self.deconv4(x)), 0.1))
        x = self.drop3_deconv(F.leaky_relu(self.dbn3(self.deconv3(x)), 0.1))
        x = self.drop2_deconv(F.leaky_relu(self.dbn2(self.deconv2(x)), 0.1))
        x = torch.tanh(self.deconv1(
            x))  # Images are normalized to -1,1 range hence use tanh. Remove batchnorm because it should fit the final distribution
        return x.view(-1, self.frames, 3, 64, 64)  # Convert the stacked batches back into frames. Images are 64*64*3

    def encode_z(self, x):
        lstm_out, _ = self.z_lstm(x)
        lstm_out, _ = self.z_rnn(lstm_out)
        post_z_list = []
        prior_z_lost = []
        zt_obs_list = []
        zt_1_mean = self.z_mean(lstm_out[:,0])
        zt_1_lar = self.z_logvar(lstm_out[:,0])

        post_z_1 = Normal(zt_1_mean, F.softplus(zt_1_lar) + 1e-5)

        post_z_list.append(post_z_1)
        prior_z0 = torch.distributions.Normal(torch.zeros(self.z_dim).to(self.device),
                                              torch.ones(self.z_dim).to(self.device))

        prior_z_lost.append(prior_z0)
        # decode z0 observation
        zt_1_dec = post_z_1.rsample()

        zt_obs_list.append(zt_1_dec)
        batch_size = lstm_out.shape[0]
        seq_size = lstm_out.shape[1]
        each_block_size = self.z_dim // self.block_size

        zt_1 = [prior_z0.rsample() for i in range(batch_size)]
        zt_1 = torch.stack(zt_1, dim=0)
        # init wt
        wt = torch.ones(batch_size, self.block_size).to(device)
        z_fwd_list = [torch.zeros(batch_size, each_block_size).to(device) for i in range(self.block_size)]
        for t in range(1, seq_size):

            if torch.isnan(zt_1).any().item():
                print('zt-1 in process is nan and sequence num is %d'%(t))

            # posterior over ct, q(ct|ot,ft)
            zt_post_mean = self.z_mean(lstm_out[:, t])
            zt_post_lar = self.z_logvar(lstm_out[:, t])

            z_post = Normal(zt_post_mean, F.softplus(zt_post_lar) + 1e-5)

            post_z_list.append(z_post)
            # p(xt|zt)
            zt_obs_list.append(z_post.rsample())


            for fwd_t in range(self.block_size):
                # prior over ct of each block, ct_i~p(ct_i|zt-1_i)
                if fwd_t==0:
                    zt_1_tmp = concat(zt_1[:, fwd_t * each_block_size:(fwd_t + 2) * each_block_size], torch.zeros(batch_size, self.z_dim//self.block_size).to(device))
                if fwd_t==1:
                    zt_1_tmp = zt_1
                if fwd_t==2:
                    zt_1_tmp = concat(torch.zeros(batch_size, self.z_dim//self.block_size).to(device),zt_1[:, (fwd_t-1) * each_block_size:(fwd_t + 1) * each_block_size])

                z_fwd_list[fwd_t] = self.z_to_c_fwd_list[fwd_t](zt_1_tmp, z_fwd_list[fwd_t],w=wt[:,fwd_t].view(-1,1))

            z_fwd_all = torch.stack(z_fwd_list, dim=2).view(batch_size, self.z_dim)
            # update weight, w0<...<wd<=1, d means block_size
            wt = self.z_w_function(z_fwd_all)
            wt = cumsoftmax(wt)

            zt_prior_mean = self.z_mean_prior(z_fwd_all)
            zt_prior_lar = self.z_logvar_prior(z_fwd_all)

            # store the prior of ct_i
            z_prior = Normal(zt_prior_mean, F.softplus(zt_prior_lar) + 1e-5)
            prior_z_lost.append(z_prior)

            zt = z_prior.rsample()

            # decode observation
            zt_1 = zt

        zt_obs_list = torch.stack(zt_obs_list, dim=1)

        return post_z_list, prior_z_lost, zt_obs_list

    def forward(self, x):
        conv_x = self.encode_frames(x)

        post_zt, prior_zt, z = self.encode_z(conv_x)
        recon_x = self.decode_frames(z)
        return post_zt, prior_zt, z, recon_x

def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


def _kl_normal_normal(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

def loss_fn(original_seq, recon_seq, post_z, prior_z):
    mse = []
    for i in range(recon_seq.shape[0]):
        mse.append(F.mse_loss(recon_seq[i], original_seq[i], reduction='sum'))
    mse = torch.stack(mse)
    # compute kl related to states, kl(q(ct|ot,ft)||p(ct|zt-1)) and kl(q(z0|f0)||N(0,1))
    kl_z_list = []
    for t in range(len(post_z)):
        if torch.isnan(post_z[t].mean).any().item() or torch.isnan(post_z[t].scale).any().item():
            print('-------------------------* %d *' % (t))
            print('ct posterior is nan')
        if torch.isnan(prior_z[t].mean).any().item() or torch.isnan(prior_z[t].scale).any().item():
            print('-------------------------* %d *' % (t))
            print('ct prior is nan')
        # kl divergences (sum over dimension)
        kl_obs_state = _kl_normal_normal(prior_z[t], post_z[t]).sum(-1).mean()
        kl_z_list.append(kl_obs_state)
    kld_z = torch.stack(kl_z_list)
    mse_loss = mse.mean()
    kl_loss = kld_z.sum()

    return mse_loss + kl_loss, mse_loss.item(), kl_loss.item()

class Trainer(object):
    def __init__(self, model, device, train, test, trainloader, testloader, epochs, batch_size, learning_rate, nsamples,
                 sample_path, recon_path, checkpoints, log_path, grad_clip):
        self.trainloader = trainloader
        self.train = train
        self.test = test
        self.testloader = testloader
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.grad_clip = grad_clip
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.samples = nsamples
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.log_path = log_path
        self.test_z = torch.randn(self.samples, model.frames, model.z_dim, device=self.device)

        self.epoch_losses = []

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.epoch_losses},
            self.checkpoints)

    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def sample_frames(self, epoch):
        with torch.no_grad():
            each_block_size = self.model.z_dim // self.model.block_size
            zt_dec = []
            prior_z0 = torch.distributions.Normal(torch.zeros(self.model.z_dim).to(self.device),
                                                  torch.ones(self.model.z_dim).to(self.device))

            zt_1 = [prior_z0.rsample() for i in range(self.samples)]
            zt_1 = torch.stack(zt_1, dim=0)
            zt_dec.append(zt_1)
            # init wt
            wt = torch.ones(self.samples, self.model.block_size).to(device)
            z_fwd_list = [torch.zeros(self.samples, each_block_size).to(device) for i in range(self.model.block_size)]
            for t in range(1, self.model.frames):


                for fwd_t in range(self.model.block_size):
                    # prior over ct of each block, ct_i~p(ct_i|zt-1_i)
                    if fwd_t == 0:
                        zt_1_tmp = concat(zt_1[:, fwd_t * each_block_size:(fwd_t + 2) * each_block_size],
                                          torch.zeros(self.samples, self.model.z_dim // self.model.block_size).to(device))
                    if fwd_t == 1:
                        zt_1_tmp = zt_1
                    if fwd_t == 2:
                        zt_1_tmp = concat(torch.zeros(self.samples, self.model.z_dim//self.model.block_size).to(device),
                                          zt_1[:, (fwd_t - 1) * each_block_size:(fwd_t + 1) * each_block_size])

                    z_fwd_list[fwd_t] = self.model.z_to_c_fwd_list[fwd_t](zt_1_tmp, z_fwd_list[fwd_t], wt[:, fwd_t].view(-1,1))

                z_fwd_all = torch.stack(z_fwd_list, dim=2).view(self.samples, self.model.z_dim)
                # update weight, w0<...<wd<=1, d means block_size
                wt = self.model.z_w_function(z_fwd_all)
                wt = cumsoftmax(wt)

                zt_prior_mean = self.model.z_mean_prior(z_fwd_all)
                zt_prior_lar = self.model.z_logvar_prior(z_fwd_all)

                # store the prior of ct_i
                z_prior = Normal(zt_prior_mean, F.softplus(zt_prior_lar) + 1e-5)
                zt = z_prior.rsample()
                zt_dec.append(zt)

                # decode observation
                zt_1 = zt

            zt_dec = torch.stack(zt_dec, dim=1)
            recon_x = self.model.decode_frames(zt_dec)
            recon_x = recon_x.view(16, 3, 64, 64)
            torchvision.utils.save_image(recon_x, '%s/epoch%d.png' % (self.sample_path, epoch))

    def recon_frame(self, epoch, original):
        with torch.no_grad():
            _, _, _, recon = self.model(original)
            image = torch.cat((original, recon), dim=0)
            print(image.shape)
            image = image.view(16, 3, 64, 64)
            torchvision.utils.save_image(image, '%s/epoch%d.png' % (self.recon_path, epoch))

    def train_model(self):
        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            write_log("Running Epoch : {}".format(epoch + 1), self.log_path)
            for i, data in enumerate(self.trainloader, 1):
                data = data.to(device)
                self.optimizer.zero_grad()
                post_z, prior_z, z, recon_x = self.model(data)
                loss, mse, kl = loss_fn(data, recon_x, post_z, prior_z)
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                loss_info = 'mse loss is %f, kl loss is %f' % (mse, kl)
                print('mse loss is %f, kl loss is %f' % (mse, kl))
                write_log(loss_info, self.log_path)

            meanloss = np.mean(losses)
            self.epoch_losses.append(meanloss)
            write_log("Epoch {} : Average Loss: {}".format(epoch + 1, meanloss), self.log_path)
            self.save_checkpoint(epoch)
            self.model.eval()
            self.sample_frames(epoch + 1)
            sample = self.test[int(torch.randint(0, len(self.test), (1,)).item())]
            sample = torch.unsqueeze(sample, 0)
            sample = sample.to(self.device)
            self.recon_frame(epoch + 1, sample)
            self.model.train()
        print("Training is complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hybrid_vae")
    parser.add_argument('--seed', type=int, default=111)
    # method
    parser.add_argument('--method', type=str, default='Hybrid')
    # dataset
    parser.add_argument('--dset_name', type=str, default='moving_mnist')
    # state size
    parser.add_argument('--z-dim', type=int, default=36)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--conv-dim', type=int, default=1024)
    # data size
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--frame-size', type=int, default=8)
    parser.add_argument('--nsamples', type=int, default=2)

    # optimization
    parser.add_argument('--learn-rate', type=float, default=0.0001)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--gpu_id', type=int, default=0)

    FLAGS = parser.parse_args()
    device = torch.device('cuda:%d' % (FLAGS.gpu_id) if torch.cuda.is_available() else 'cpu')

    vae = FullQDisentangledVAE(frames=FLAGS.frame_size, z_dim=FLAGS.z_dim, hidden_dim=FLAGS.hidden_dim, conv_dim=FLAGS.conv_dim, device=device)
    sprites_train = Sprites('./dataset/lpc-dataset/train/', 6687)
    sprites_test = Sprites('./dataset/lpc-dataset/test/', 873)
    starttime = datetime.datetime.now()
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    time_dir = now.strftime('%Y_%m_%d_%H_%M_%S')
    base_path = './%s/%s'%(FLAGS.method, time_dir)
    model_path = '%s/model' % (base_path)
    log_recon = '%s/recon' % (base_path)
    log_sample = '%s/sample' % (base_path)
    log_path = '%s/log_info.txt' % (base_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_recon):
        os.makedirs(log_recon)
    if not os.path.exists(log_sample):
        os.makedirs(log_sample)

    write_log(FLAGS, log_path)

    trainloader = torch.utils.data.DataLoader(sprites_train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(sprites_test, batch_size=1, shuffle=True, num_workers=4)

    trainer = Trainer(vae, device, sprites_train, sprites_test, trainloader, testloader, epochs=FLAGS.max_epochs, batch_size=FLAGS.batch_size,
                      learning_rate=FLAGS.learn_rate, checkpoints='%s/%s-disentangled-vae.model'%(model_path, FLAGS.method), nsamples=FLAGS.nsamples,
                      sample_path=log_sample,
                      recon_path=log_recon, log_path = log_path, grad_clip=FLAGS.grad_clip)
    trainer.load_checkpoint()
    trainer.train_model()
    endtime = datetime.datetime.now()
    seconds = (endtime - starttime).seconds
    hours = seconds // 3600
    minutes = (seconds % 3600)//60
    second = (seconds % 3600)%60
    print((endtime - starttime))
    timeStr = "running time: " + str(hours)+ 'hours'+ str(minutes) + 'minutes' + str(second) + "second"
    write_log(timeStr, log_path)
