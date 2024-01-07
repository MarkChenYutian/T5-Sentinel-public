import torch
import click
from pathlib import Path
from transformers import T5TokenizerFast as Tokenizer
from detector.t5_sentinel.model import Sentinel


class SentinelGradientExtractor(torch.nn.Module):
    def __init__(self, embedder, interpolate_step=100) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder_mode = True

        self.max_step = interpolate_step
        self.step = 0
        self.tokens = None
        self.embed_result = None
        self.pad_result = None
        self.mid_results = []

    def reset(self):
        self.encoder_mode = True
        self.step = 0
        self.embed_result = None
        self.pad_result = None
        self.mid_results = []

    def pure_forward(self, *args, **kwargs):
        return self.embedder(*args, **kwargs)

    def grad_forward(self, *args, **kwargs):
        # print(f"Gradient integrating - Step {self.step} / {self.max_step}")
        if self.step == 0:
            embedding = self.embedder(*args, **kwargs)
            self.tokens = args[0]
            self.pad_result = self.embedder(torch.zeros_like(args[0])).detach()
            self.embed_result = embedding.detach()

        embed_result = self.embed_result.clone()
        embed_result.requires_grad_(True)
        embed_result.retain_grad()
        self.mid_results.append(embed_result)

        mix_percent = self.step / self.max_step
        mix_result: torch.Tensor = (
            mix_percent * embed_result + (1 - mix_percent) * self.pad_result
        )
        # mix_result.requires_grad_(True)
        # mix_result.retain_grad()
        # self.mid_results.append(mix_result)
        self.step += 1

        return mix_result

    def forward(self, *args, **kwargs):
        # Embedding layer will be called twice by T5, the first call is for encoder
        # second call is for decoder
        if self.encoder_mode:
            self.encoder_mode = False
            return self.grad_forward(*args, **kwargs)
        else:
            self.encoder_mode = True
            return self.pure_forward(*args, **kwargs)


def injectSentinel(model: Sentinel):
    model.injected_embedder = None

    def auto_embedder_substitution(*args, **kwargs):
        embedder = model.backbone.get_input_embeddings()
        grad_embedder = SentinelGradientExtractor(embedder, 100)
        model.injected_embedder = grad_embedder
        model.backbone.set_input_embeddings(grad_embedder)
        print("Embedder substitution Complete.")

    model.register_load_state_dict_post_hook(auto_embedder_substitution)
    return model


model = injectSentinel(Sentinel())
checkpoint = torch.load(Path("data", "checkpoint", "T5Sentinel.0613.pt"))
model.load_state_dict(checkpoint["model"])
model = model.cuda().eval()


def explain(text, label):
    tokenizer = Tokenizer.from_pretrained("t5-small", model_max_length=512)
    label_tokenizer = Tokenizer.from_pretrained("t5-small", model_max_length=2)

    text_tokenized = tokenizer.batch_encode_plus(
        (text,), padding=True, truncation=True, return_tensors="pt"
    )
    lab_tokenized = label_tokenizer.batch_encode_plus(
        (label,), padding=True, truncation=True, return_tensors="pt"
    )

    for i in range(100):
        prob = model.interpretability_study_entry(
            text_tokenized.input_ids.cuda(),
            text_tokenized.attention_mask.cuda(),
            lab_tokenized.input_ids.cuda(),
        )

    all_gradient = [mid.grad for mid in model.injected_embedder.mid_results]
    avg_gradient = torch.zeros_like(all_gradient[0])
    for i in range(len(all_gradient)):
        avg_gradient += all_gradient[i]
    avg_gradient = avg_gradient / len(all_gradient)
    diff = model.injected_embedder.embed_result - model.injected_embedder.pad_result
    integrated_gradient = torch.norm((avg_gradient * diff)[0], dim=1)

    pred_label = ["Human", "ChatGPT", "PaLM", "LLaMA", "GPT2"]
    pred_idx = torch.argmax(prob).item()
    print(f"Predicted as {pred_label[pred_idx]} with prob of {prob[0, pred_idx]}")

    tokens = [tokenizer.decode(tok) for tok in model.injected_embedder.tokens[0]]
    model.injected_embedder.reset()
    return integrated_gradient, tokens, prob


def visualize_explain_simple(text, label):
    gradient, tokens, prob = explain(text, label)
    avg_grad = torch.mean(gradient).item()
    std_grad = torch.std(gradient).item()
    more_than_1std = gradient > avg_grad + 1 * std_grad
    more_than_0std = gradient > avg_grad
    mask_0std = torch.logical_xor(more_than_1std, more_than_0std)
    mask_1std = more_than_1std
    for idx in range(gradient.shape[0]):
        tok = tokens[idx]
        if mask_0std[idx].item():
            print(click.style(tok, fg="yellow"), end=" ")
        elif mask_1std[idx].item():
            print(click.style(tok, fg="red"), end=" ")
        else:
            print(tok, end=" ")
    print("")


if __name__ == "__main__":
    visualize_explain_simple("Hello world!", "<extra_id_0>")
    visualize_explain_simple(
        'Media playback is unsupported on your device Media caption Hungarian Prime Minister Viktor Orban: "It\'s a serious ecological catastrophe"\n\nToxic red sludge from a spill at an industrial plant in Hungary has reached the River Danube, officials say.\n\nThey said alkaline levels that killed all fish in one river were now greatly reduced, but were being monitored.\n\nPM Viktor Orban called the spill an "ecological tragedy". There are fears the mud, which burst out of a reservoir on Monday, could poison the Danube.\n\nCountries downstream from Hungary, including Croatia, Serbia and Romania, are drawing up emergency plans.\n\nA million cubic metres (35m cu ft) of the sludge spilled from a reservoir at an alumina plant in Ajka in western Hungary. Four people were killed and about 100 injured.\n\nThe mud also caused massive damage in nearby villages and towns, as well as a wide swathe of farmland.\n\nNo victory declaration\n\nDisaster official Tibor Dobson said all life in the Marcal river, which feeds the Danube, had been "extinguished".\n\nThe BBC\'s Nick Thorpe in western Hungary says news that the spill has now reached the Danube is worrying.\n\nTests are being carried out for two potential hazards - a powerful alkaline solution and heavy metals.\n\nOfficials say both are below toxic levels for humans in the Danube and its tributary, the Raba.\n\nBut Mr Dobson said this was "by no means a victory declaration".\n\nDead fish have been spotted in both rivers, Mr Dobson notes.\n\nTo save their eco-system, he adds, pH levels must be reduced to 8 from about 9 recently recorded at the confluence of the Raba with the Danube.\n\nThe authorities have been pouring a mixture of clay and acid to reduce alkalinity.\n\n"The main effort is now being concentrated on the Raba and the Danube," Mr Dobson said. "That\'s what has to be saved."\n\nPhilip Weller, executive director of the International Commission for the Protection of the Danube, told the BBC that that the best one could hope was for the Danube to dilute the toxic sludge.\n\n"It\'s a rather large amount of water in the Danube that the dilution effects will at least mean that there will not be immediate consequences," he said.\n\nAbandoning villages\n\nEnvironmental expert Paul Younger of Newcastle University says high alkaline concentrations are an irritant, but not life-threatening for people.\n\n"It\'s not like a big cyanide spill," he told the BBC.\n\nThe sludge itself is a hazardous mixture of water and mining waste containing heavy metals.\n\nThe victims are believed to have drowned, with the depth of the fast-moving flood reaching 2m (6.5ft) in places, but many of those injured suffered chemical burns.\n\nOn Thursday Mr Orban visited the village of Kolontar, the worst-affected settlement, and said some areas would have to be abandoned.\n\n"Hungary is strong enough to be able to combat the effects of such a catastrophe. But we\'re still open to any expertise which will help us combat the pollution effects," he added.\n\nAngry villagers confronted a company official in Kolontar on Wednesday evening. They say they plan to sue the firm for damages.\n\nHerwit Schuster, a spokesman for Greenpeace International, described the spill as "one of the top three environmental disasters in Europe in the last 20 or 30 years".\n\nLand had been "polluted and destroyed for a long time", he told AP.\n\n"If there are substances like arsenic and mercury, that would affect river systems and ground water on long-term basis," he added.',
        "<extra_id_0>",
    )
