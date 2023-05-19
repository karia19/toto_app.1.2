# Server for toto_app.1.2 frontend
### All files are docker and to run docker go folder server
```
docker compose up -d
```
### To run crontab
```
docker exec CONATAINER_ID bin/bash

service cron start
```


## Known Issues

- Some track statistics may not work as expected.
- The odds data in the historical odds endpoint may be missing for certain days.
- The database has not been updated since September 2022.

## Contributing

Contributions are welcome! If you find any bugs, have feature requests, or want to contribute improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
