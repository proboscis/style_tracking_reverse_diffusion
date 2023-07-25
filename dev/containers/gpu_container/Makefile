
#STRD_ROOT='$(HOME)/repos/style_tracking_reverse_diffusion_impl'

copy_sources:
	rsync -avh --progress --exclude=dev --exclude=.git $(STRD_ROOT)/ $(STRD_ROOT)/dev/containers/gpu_container/sources/

container: copy_sources
	docker build -t proboscis/strd .

run_transfer: container
	docker run --rm --gpus all --shm-size=10.24gb -v /home/.cache:/root/.cache -v /home:/root/host_home \
	$(STRD_CONTAINER_NAME) python examples/main.py \
	transfer_style \
	--style example_data/image_pairs/style_0.png --content example_data/image_pairs/content_0.png \
	- to '"image,RGB,RGB"' - save '/root/host_home/result.png'

container_bash: container
	docker run -it --rm --gpus all $(STRD_CONTAINER_NAME) bash

webapp: container
	docker run --rm --gpus all --shm-size=10.24gb -v /home/.cache:/root/.cache -v /home:/root/host_home --net=host \
	$(STRD_CONTAINER_NAME) python examples/web_demo.py \
	web_app


push_container: container
	docker push $(STRD_CONTAINER_NAME)
