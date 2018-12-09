import time, copy, torch

def train_model(model, dataloader, criterion, optimiser, scheduler, device, dataset_size, num_epochs=10):
    try:
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            

            for phase in ['train', 'valid']:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for data in dataloader[phase]:
                    images, metadata, labels = data['image'], data['metaData'], data['label']
                    images = images.to(device)
                    metadata = matadata.to(device)
                    labels = labels.to(device)

                    optimiser.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        outputs = model((images, metadata))
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimiser.step()
                    temp1 = loss.item() * images.size(0)
                    temp2 = torch.sum(preds==labels.data)
                    running_loss += temp1
                    running_corrects += temp2
                    print('Running loss:{:.4f} Acc: {:.2f}'.format(temp1/images.size(0), temp2*100/images.size(0)), end='\r')
                epoch_loss = running_loss/dataset_size[phase]
                epoch_acc = running_corrects.double()/dataset_size[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                          epoch_acc*100))

                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    resume_model_wts = copy.deepcopy(model.state_dict())
                    best_model_wts = copy.deepcopy(model.state_dict())
            print('--' * 20)
            print()

        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(
                time_elapsed//60, time_elapsed%60))
        print('Best valid Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
    except KeyboardInterrupt:
        model.load_state_dict(best_model_wts)
        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(
                time_elapsed//60, time_elapsed%60))
        print('Best valid Acc: {:4f}'.format(best_acc))
        return model
    return model