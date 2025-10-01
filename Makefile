.PHONY: run dev stop clean 

run: 
	docker compose up --build 

dev: 
	docker compose --profile dev up
	docker exec -it tapsegnn-dev-container bash 

stop: 
	docker compose down 

clean: 
	docker compose down -v 
	docker system prune -f

clean-vc: 
	docker compose down -v