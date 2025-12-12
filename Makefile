.PHONY: test dev stop clean clean-vc  

test: 
	docker compose --profile test up --build 
dev: 
	docker compose --profile dev up -d --build
	docker exec -it tapsegnn-dev-container bash 

stop: 
	docker compose down 

clean: 
	docker compose down -v 
	docker system prune -f

clean-vc: 
	docker compose down -v
