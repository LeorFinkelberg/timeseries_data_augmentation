import typing as t
import numpy as np
import pandas as pd
import swifter


class TransformData:
    """
    The class identifies statistically significant isolated observations that
    could be mistaken for outliers and enriches the neighbourhoods of these
    isolated observations with synthetic data constructed by assuming a linear
    relationship between the ordinates of the features in the defined neighbourhood.

    1. The target time series is checked for statistically significant outliers
    using a modified z-score estimator. If there are no measurements with such
    outliers, the augmentation algorithm terminates. If there are, then step 2
    is executed for all measurements found.
    2 Determine the neighbourhood of the linear interpolation of the data.
    The right and left boundaries of the neighbourhood are bounded by the
    nearest neighbouring measurements.
    3. Augment the data in the area defined in the previous step by linear interpolation.

    """

    def __init__(
        self,
        target_column: str,
        time_column: str,
        N: int = 15,
        k: int = 10,
    ):
        """
        Args:
            target_column (str) : name of the target column
            time_column (str) : name of the time column
            N (int, optional) : frequency or intensity of augmentation (i.e.
            the number of new augmented points for every sample). Defaults to 15.
            k (int, optional) : defines the threshold for comparing modified z-score
            estimations. The higher this threshold, the fewer neighbourhoods of
            statistically significant samples will be augmented. Defaults to 10.
        """
        self.target = target_column
        self.N = N
        self.k = k
        self.time_column = time_column

    def augment_data(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Enriches the dataset with interpolation points
        in the neighbourhood of the peaks

        Args:
            df (pd.DataFrame): DataFrame for augmentation

        Returns:
            pd.DataFrame: augmented DataFrame
        """

        feature: pd.Series = df[self.target]

        k = self.k
        N = self.N

        left_right_idxs: t.List[t.Tuple[int, int, int]] = self._left_right_of_peak(
            ser=feature, k=k
        )

        ys_left_br = []
        ys_right_br = []

        for left_idx, peak_idx, right_idx in left_right_idxs:
            # left branch
            x1_left_br, x2_left_br = 0, 1
            x_step_left_br = (x2_left_br - x1_left_br) / N
            y1_left_br = feature[feature.index == left_idx].squeeze()
            y2_left_br = feature[feature.index == peak_idx].squeeze()

            xi_left_br = 0

            for step in range(1, N + 1):
                xi_left_br += x_step_left_br
                y_left_br = (y2_left_br - y1_left_br) / (x2_left_br - x1_left_br) * (
                    xi_left_br - x1_left_br
                ) + y1_left_br
                ys_left_br.append(
                    [(left_idx, step), y_left_br] + [np.nan] * (df.shape[1] - 1)
                )

            # right branch
            x1_right_br, x2_right_br = 0, 1
            x_step_right_br = (x2_right_br - x1_right_br) / N
            y1_right_br = feature[feature.index == peak_idx].squeeze()
            y2_right_br = feature[feature.index == right_idx].squeeze()

            xi_right_br = 0

            for step in range(1, N + 1):
                xi_right_br += x_step_right_br
                y_right_br = (y2_right_br - y1_right_br) / (
                    x2_right_br - x1_right_br
                ) * (xi_right_br - x1_right_br) + y1_right_br
                ys_right_br.append(
                    [(peak_idx, step), y_right_br] + [np.nan] * (df.shape[1] - 1)
                )

        df = df.reset_index()
        df["index"] = df["index"].apply(lambda elem: (elem, 0))

        df = pd.concat([df, pd.DataFrame(ys_left_br, columns=df.columns)])
        df = pd.concat([df, pd.DataFrame(ys_right_br, columns=df.columns)])

        df = df.swifter.progress_bar(False).apply(
            lambda col: pd.to_numeric(col, errors="ignore")
        )

        df = df.set_index("index").drop_duplicates().sort_index()

        time_column = df[self.time_column]
        df = df.drop(columns=[self.time_column]).interpolate(
            method="linear", limit_direction="forward", axis=0
        )
        df.insert(0, self.time_column, time_column)

        return df

    def _modified_Z_score(
        self,
        ser: pd.Series,
        consistency_correction: float = 1.4826,
    ) -> float:
        """
        calculate modified Z-evaluation
        """
        med = np.median(ser)
        dev_med = np.abs(ser - med)
        mad = np.median(np.abs(dev_med))
        mod_Z_sore = dev_med / (consistency_correction * mad)

        return mod_Z_sore

    def _left_right_of_peak(
        self, ser: pd.Series, k: int = 10
    ) -> t.List[t.Tuple[int, int, int]]:
        """
        Finds the closest indexes from the left and right of the peak index

        Args:
            ser (pd.Series): Series
            k (int, optional): defines the threshold for comparing modified z-score
            estimations. The higher this threshold, the fewer neighbourhoods of
            statistically significant samples will be augmented. Defaults to 10.

        Returns:
            t.List[t.Tuple[int, int, int]]: list of 3-tuples with the nearest indexes
             from left and right from the current peak
        """
        mod_z_score = self._modified_Z_score(ser)
        outlier_idxs = ser[mod_z_score > k * np.median(mod_z_score)].index.to_list()

        # list of 3-tuples with the nearest indexes from left and right from current peak
        left_right_idxs = []

        for idx in outlier_idxs:
            # searching from the nearest index from the left
            idx_left = idx - 1
            while not (ser.index == idx_left).sum():
                if idx_left < ser.index.min():
                    # idx_left += 1 # todo
                    idx_left = ser.index.min()
                    break
                idx_left -= 1

            # searching for nearest index from the right
            idx_right = idx + 1
            while not (ser.index == idx_right).sum():
                if idx_right > ser.index.max():
                    # idx_right -= 1 # todo
                    idx_right = ser.index.max()
                    break
                idx_right += 1

            left_right_idxs.append((idx_left, idx, idx_right))

        return left_right_idxs
