.PHONY: run dev stop clean 

run: 
	docker compose up -d --build 

dev: 
	docker compose --profile dev up -d 
	docker exec -it tapsegnn-dev-container bash 

stop: 
	docker compose down 

clean: 
	docker compose down -v 
	docker system prune -f