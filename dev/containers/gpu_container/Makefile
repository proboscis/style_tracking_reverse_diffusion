
ROOT='$(HOME)/repos/style_tracking_reverse_diffusion_impl'

copy_sources:
	rsync -avh --progress --exclude=dev --exclude=.git $(ROOT)/ $(ROOT)/dev/containers/gpu_container/sources/

container: copy_sources
	docker build -t proboscis/strd .

run_transfer: container
	docker run --rm --gpus all --shm-size=10.24gb -v /home/.cache:/root/.cache -v /home:/root/host_home \
	proboscis/strd:latest python examples/main.py \
	transfer_style \
	--style example_data/image_pairs/style_0.png --content example_data/image_pairs/content_0.png \
	- to '"image,RGB,RGB"' - save '/root/host_home/result.png'

container_bash: container
	docker run -it --rm --gpus all proboscis/strd:latest bash

webapp: container
	docker run --rm --gpus all --shm-size=10.24gb -v /home/.cache:/root/.cache -v /home:/root/host_home --net=host \
	proboscis/strd:latest python examples/web_demo.py \
	web_app


push_container: container
	docker push proboscis/strd:latest
