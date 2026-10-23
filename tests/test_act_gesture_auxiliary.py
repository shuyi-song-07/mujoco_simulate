import torch

from training.ACT.train_act_variation_gesture import (
    masked_gesture_loss, masked_hand_delta_loss,
)


def test_masked_gesture_loss_ignores_unknown_and_padded_frames():
    logits = torch.zeros(1, 4, 7, requires_grad=True)
    labels = torch.tensor([[[1], [0], [7], [2]]])
    action_pad = torch.tensor([[False, False, False, True]])
    gesture_pad = torch.tensor([[False, False, False, False]])

    loss, count = masked_gesture_loss(logits, labels, action_pad, gesture_pad)

    assert count == 2
    assert torch.isclose(loss, torch.log(torch.tensor(7.0)))
    loss.backward()
    assert logits.grad is not None


def test_masked_hand_delta_loss_uses_only_valid_finite_real_frames():
    prediction = torch.zeros(1, 5, 2, requires_grad=True)
    target = torch.tensor([[[0.03, -0.06], [0.3, 0.3], [float("nan"), 0.0],
                            [0.03, 0.03], [0.03, 0.03]]])
    valid = torch.tensor([[[1], [0], [1], [1], [1]]])
    action_pad = torch.tensor([[False, False, False, True, False]])
    delta_pad = torch.tensor([[False, False, False, False, True]])
    valid_pad = torch.zeros(1, 5, dtype=torch.bool)

    loss, count = masked_hand_delta_loss(
        prediction, target, valid, action_pad, delta_pad, valid_pad, scale=0.03,
    )

    assert count == 1
    # Smooth-L1(0, 1) = 0.5 and Smooth-L1(0, -2) = 1.5; mean = 1.0.
    assert torch.isclose(loss, torch.tensor(1.0))
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[0, 1:]) == 0
