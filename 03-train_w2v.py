import torch
import models
import dataset
import datetime
import pickle
import wandb
import tqdm


torch.manual_seed(42)
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# Load the tokeniser
with open('./corpus/tokeniser.pkl', 'rb') as f:
    tokeniser = pickle.load(f)
    
words_to_ids = tokeniser['words_to_ids']
ids_to_words = tokeniser['ids_to_words']

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w2v = models.SkipGram(voc=len(words_to_ids), emb=128).to(dev) # initalise the model & send to device
torch.save(w2v.state_dict(), f'./checkpoints/{ts}.0.w2v.pth') # save the initial model
print('w2v:', sum(p.numel() for p in w2v.parameters())) # 35,998,976
opt = torch.optim.Adam(w2v.parameters(), lr=0.003)
wandb.init(project='mlx6-02-w2v')

# Load the dataset
ds = dataset.Window('./corpus/tokens.txt')
dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)

for epoch in range(5):
  prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
  for idx, (inpt, trgs) in enumerate(prgs):
    inpt, trgs = inpt.to(dev), trgs.to(dev)
    rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2)).to(dev)
    opt.zero_grad()
    loss = w2v(inpt, trgs, rand)
    loss.backward()
    opt.step()
    wandb.log({'loss': loss.item()})
    if idx % 10_000 == 0: torch.save(w2v.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.w2v.pth')


