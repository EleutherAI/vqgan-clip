class BoxCropper(object): 
    def __init__(self, w=0.3, h=0.3):
      self.w, self.h = w, h

    def sample(self, source):
        w, h = int(source.width*self.w), int(source.height*self.h)
        w, h = torch.randint(w//2, w+1, []).item(), torch.randint(h//2, h+1, []).item()
        h = w
        x1 = torch.randint(0, source.width - w + 1, []).item()
        y1 = torch.randint(0, source.height - h + 1, []).item()
        x2, y2 = x1 + w, y1 + h
        box = x1, y1, x2, y2
        crop = source.crop(box)
        mask = torch.zeros([source.size[1], source.size[0]])
        mask[y1:y2, x1:x2] = 1.
        return crop, mask

def sample(source, sampler, model, preprocess, n=64000, batch_size=128):
    n_batches = 0- -n // batch_size  # round up
    t_crop = 0

    model.eval()
    with torch.no_grad():
        for step in tqdm(range(n_batches)):
            t_crop = float(step)/float(n_batches)
            crop_cur = (0.4) * (1- t_crop) + (0.1) * t_crop
            sampler.w = crop_cur
            sampler.h = crop_cur

            batch = []
            for _ in range(batch_size):
                crop, mask = sampler.sample(source)
                batch.append((preprocess(crop).unsqueeze(0).to(next(model.parameters()).device), mask))
            crops = torch.cat([img for img, *_ in batch], axis=0)
            embeddings = model.encode_image(crops).cpu().detach()

            for emb, msk in zip(embeddings, [mask for _, mask, *_ in batch]):
                yield emb, msk
    # return samples

def aggregate(samples, labels, model):
    texts = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(texts).cpu()
    masks = []
    for label, text_emb in zip(labels, text_embeddings):
        text_features = text_emb / text_emb.norm(dim=-1, keepdim=True)
        pixel_sum = torch.ones_like(next(samples)[1])
        samples_per_pixel = torch.ones_like(next(samples)[1])
        # dists = [spherical_dist(text_emb.float(), embedding.float()).item()
        #          for embedding, *_ in samples]
        # min_dist, max_dist = min(dists), max(dists)
        for embedding, mask in samples: # dist, (embedding, mask) in zip(dists, samples):
            image_features = embedding / embedding.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp().to(image_features.device)
            logits_per_image = logit_scale * image_features @ text_features.t()
            dist = logits_per_image.float().exp().item()
            # dist = spherical_dist(text_emb.float(), embedding.float()).item()
            pixel_sum += mask * dist
            samples_per_pixel += mask
        img = pixel_sum / samples_per_pixel
        # img *= 4
        # print(img.max())
        # print(img.min(), img.max())
        img = ((img - img.min()
        ) / img.max()) ** 2 # 0.75
        # img /= img.max()
        #img[img <= 0.001] = 0.
        masks.append((img, label))
    return masks

def visualise(source, masks):
    source = TF.to_tensor(source)
    for img, label in masks:
        TF.to_pil_image(source * img[None]).save('mask_temp.png')
        display.display(display.Image('mask_temp.png'))

def save(masks):
    source = torch.ones_like(masks[0])
    for img, label in masks:
        return source * img[None]

