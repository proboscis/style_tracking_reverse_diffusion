
install:
	pip install -e .
	cd ../; git clone https://github.com/CompVis/stable-diffusion.git
	cd ../stable-diffusion; pip install -e .
	wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O sd-v1-4.ckpt

