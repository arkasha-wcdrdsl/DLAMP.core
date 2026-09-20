from datetime import datetime, timedelta

import numpy as np
from torch import Size


class TimeUtil:
    @staticmethod
    def entire_period(
        year: int,
        month: int,
        day: int,
        hour: int | None = None,
        interval: dict[str, int] | timedelta = {"minutes": 1},
    ) -> list[datetime]:
        """
        Generate a list of datetime objects representing the entire period for a given date and time interval.

        Parameters:
            year (int): The year of the target date.
            month (int): The month of the target date.
            day (int): The day of the target date.
            hour (int | None, optional): The hour of the target date. Defaults to None.
            interval (dict[str, int] | timedelta, optional): The time interval used to iterate through the target dates.
                The keys of the dictionary must be one of the following: "days", "seconds", "microseconds",
                "milliseconds", "minutes", "hours", "weeks". Defaults to {"minutes": 1}.

        Returns:
            list[datetime]: A list of datetime objects representing the entire period for the given date and time interval.
        """
        if isinstance(interval, dict):
            interval = timedelta(**interval)
        time_list = []
        if hour:
            dt = datetime(year, month, day, hour)
            while dt.hour == hour:
                time_list.append(dt)
                dt += interval
        else:
            dt = datetime(year, month, day)
            while dt.day == day:
                time_list.append(dt)
                dt += interval
        return time_list

    @staticmethod
    def N_days_time_list(
        year: int,
        month: int,
        day: int,
        interval: dict[str, int] | timedelta,
        n_days: int,
    ) -> list[datetime]:
        """
        Generate a list of datetime objects representing the three days before and after a given date.

        Parameters:
            year (int): The year of the target date.
            month (int): The month of the target date.
            day (int): The day of the target date.
            interval (dict[str, int] | timedelta): The time interval used to iterate through the target dates.
                The keys of the dictionary must be one of the following: "days", "seconds", "microseconds",
                "milliseconds", "minutes", "hours", "weeks". Defaults to {"minutes": 1}.

        Returns:
            list[datetime]: A list of datetime objects representing the three days before and after the given date.
        """
        assert n_days >= 1, f"n_days must be a positive integer but get {n_days}"
        half_range = n_days // 2
        start = -half_range if n_days > 1 else 0
        end = half_range + 1 if n_days % 2 == 1 else half_range

        target_t = [
            datetime(year, month, day) + i * timedelta(days=1)
            for i in range(start, end)
        ]

        time_list = []
        for calendar in target_t:
            time_list.extend(
                TimeUtil.entire_period(
                    calendar.year, calendar.month, calendar.day, interval=interval
                )
            )
        return time_list

    @staticmethod
    def create_time_features(
        time_of_batches: list[datetime], img_shape: tuple[int, int]|Size
    ) -> np.ndarray:
        """
        Computes the sine and cosine transformations of the Day of Year (DoY) and Time
        of Day (ToD) from the provided datetime object and returns arrays of the specified
        shape filled with these values.

        Parameters:
        - time_of_batches (tuple[datetime, ...]): The datetime objects for which to compute DoY and ToD.
        - array_shape (tuple): The desired 2D shape of the output arrays.

        Returns:
        - time_features (np.ndarray): Array of shape (batch_size, **array_shape, 4) filled with the
            computed DoY and ToD values.
        """
        assert len(img_shape) == 2, "array_shape must be a tuple of length 2"

        buffer = []

        for curr_dt in time_of_batches:
            # Calculate the total number of days in the year
            total_days = datetime(curr_dt.year,12,31).timetuple().tm_yday

            # Calculate Day of Year (DoY)
            doy = curr_dt.timetuple().tm_yday

            # Calculate Time of Day (ToD) in hours
            tod = curr_dt.hour + curr_dt.minute / 60 + curr_dt.second / 3600

            # Compute sine and cosine transformations for DoY
            doy_sin = np.sin(2 * np.pi * doy / total_days)
            doy_cos = np.cos(2 * np.pi * doy / total_days)

            # Compute sine and cosine transformations for ToD
            tod_sin = np.sin(2 * np.pi * tod / 24.0)
            tod_cos = np.cos(2 * np.pi * tod / 24.0)

            # Create arrays filled with the computed values
            doy_sin_array = np.full(img_shape, doy_sin, dtype=np.float32)
            doy_cos_array = np.full(img_shape, doy_cos, dtype=np.float32)
            tod_sin_array = np.full(img_shape, tod_sin, dtype=np.float32)
            tod_cos_array = np.full(img_shape, tod_cos, dtype=np.float32)

            # stack the arrays (H, W, 4)
            time_features = np.stack(
                [doy_sin_array, doy_cos_array, tod_sin_array, tod_cos_array], axis=-1
            )

            buffer.append(time_features)

        time_features = np.stack(buffer, axis=0)

        time_features = np.broadcast_to(time_features[:,None,...], [len(time_of_batches), 1, *img_shape, 4])

        return time_features

    @staticmethod
    def recover_datetime_from_time_features(
        time_features: np.ndarray, reference_year: int = 2024
    ) -> list[datetime]:
        """
        Recover the datetime object from the provided time features array.

        Parameters:
        - time_features (np.ndarray): Array of shape (..., 4) containing the sine and cosine
            transformations of Day of Year (DoY) and Time of Day (ToD).
        - reference_year (int): The reference year to use for recovering the date. Defaults to 2024.

        Returns:
        - tuple[datetime]: A tuple contains recovered datetime object, which size equals to batch size.
        """
        assert (
            time_features.shape[-1] == 4
        ), f"time_features must have shape (..., 4): got {time_features.shape}"

        if len(time_features.shape) < 3:
            time_features = np.expand_dims(time_features, axis=0)

        res: list[datetime] = []

        for i in range(time_features.shape[0]):
            # Extract the first element's features
            doy_sin, doy_cos, tod_sin, tod_cos = time_features[i].reshape(-1, 4)[0]

            # Recover DoY and ToD using arctangent2
            doy = int(np.round((np.arctan2(doy_sin, doy_cos) * 365.0) / (2 * np.pi)))
            tod = (np.arctan2(tod_sin, tod_cos) * 24.0) / (2 * np.pi)

            # Adjust DoY to be within valid range
            if doy <= 0:
                doy += 365

            # Recover month and day from DoY
            recovered_date = datetime(reference_year, 1, 1) + timedelta(days=doy-1)

            # Recover hour, minute, second from ToD
            if tod < 0:
                tod += 24
            hour = int(tod)
            minute = int((tod - hour) * 60)
            second = int(((tod - hour) * 60 - minute) * 60)

            # Create the recovered datetime object
            res.append(recovered_date + timedelta(hours=hour,minutes=minute,seconds=second))

        return res