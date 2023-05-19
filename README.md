# Server for Toto History Odds API

This repository contains a Flask server implementation for an API that provides historical odds data for Veikkaus Toto game. The server fetches data from the Veikkaus API and stores it in a Redis database. It also provides an endpoint for calculating statistics based on track starts, drivers, and coach success in the track.

## Installation

To install the necessary dependencies, follow these steps:

1. Clone this repository to your local machine or server:

   ```bash
   git clone https://github.com/karia19/toto_app.1.2
   ```

## Usage

Ensure that Docker is installed on your system, and then follow these steps:

1. Build the Docker image:

   ```bash
   cd server
   docker compose up -d --build .
   ```
2. To run crontab
```
docker exec CONATAINER_ID bin/bash

service cron start
```


## API Endpoints

The server provides the following API endpoints:

- `GET /api/v1/toto/history_odds`: Retrieves Toto historical odds data for a specific day from the Redis database. The data is fetched from the Veikkaus API if it's not already stored.

- `GET /api/v1/toto/history`: Calculates statistics based on track starts, drivers, and coach success in the track.

Feel free to explore and integrate these endpoints into your application as needed.



## Known Issues

- Some track statistics may not work as expected.
- The odds data in the historical odds endpoint may be missing for certain days.
- The database has not been updated since September 2022.

## Contributing

Contributions are welcome! If you find any bugs, have feature requests, or want to contribute improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
