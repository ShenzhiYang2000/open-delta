from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
  
from opendelta import AdapterModel
delta_model = AdapterModel(model)
delta_model.freeze_module(exclude=["deltas", "classifier"]) # leave the delta tuning modules and the newly initialized classification head tunable.
delta_model.log() # optional: to visualize how the `model` changes. 

# training_dataloader = get_dataloader()
# optimizer, loss_function = get_optimizer_loss_function()
# for batch in training_dataloader:
#     optimizer.zero_grad()
#     targets = batch.pop('labels')
#     outputs = model(**batch).logits
#     loss = loss_function(outputs, targets)
#     loss.backward()
#     optimizer.step()
#     print(loss)

# torch.save(model.state_dict(), "finetuned_bert.ckpt")
# delta_model.save_finetuned("finetuned_bert")