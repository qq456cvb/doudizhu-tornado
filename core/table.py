import logging
import random
from typing import List

from tornado.ioloop import IOLoop

from core.robot import AiPlayer
from handlers.protocol import Protocol as Pt
from core.predictor import Predictor
from tensorpack import *
from core.DQNModel import Model

logger = logging.getLogger('ddz')


BATCH_SIZE = 8
MAX_NUM_COMBS = 100
MAX_NUM_GROUPS = 21
ATTEN_STATE_SHAPE = 60
HIDDEN_STATE_DIM = 256 + 256 * 2 + 120
STATE_SHAPE = (MAX_NUM_COMBS, MAX_NUM_GROUPS, HIDDEN_STATE_DIM)
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 3e3
INIT_MEMORY_SIZE = MEMORY_SIZE // 10
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ  # each epoch is 100k played frames
EVAL_EPISODE = 50

NUM_ACTIONS = None
METHOD = None
MODEL_PATH = '/home/neil/PycharmProjects/doudizhu-tornado/TensorPack/MA_Hierarchical_Q/train_log/model-500000'

class Table(object):

    WAITING = 0
    PLAYING = 1
    END = 2
    CLOSED = 3

    def __init__(self, uid, room):
        self.uid = uid
        self.room = room
        self.players = [None, None, None]
        self.state = 0  # 0 waiting  1 playing 2 end 3 closed
        self.pokers: List[int] = []
        self.multiple = 1
        self.call_score = 0
        self.max_call_score = 0
        self.max_call_score_turn = 0
        self.whose_turn = 0
        self.lord_turn = 0
        self.last_shot_seat = 0
        self.out_cards = [[] for _ in range(3)]
        self.controller = None
        self.last_shot_poker = []
        self.history = [None, None, None]
        if room.allow_robot:
            IOLoop.current().call_later(0.1, self.ai_join, nth=1)

    def reset(self):
        self.out_cards = [[] for _ in range(3)]
        self.controller = None
        self.pokers: List[int] = []
        self.multiple = 1
        self.call_score = 0
        self.max_call_score = 0
        self.max_call_score_turn = 0
        self.whose_turn = random.randint(0, 2)
        self.lord_turn = 0
        self.last_shot_seat = 0
        self.last_shot_poker = []
        self.players[0].send([Pt.RSP_RESTART])
        for h in self.history:
            if h is not None:
                h.clear()
        for player in self.players:
            # player.join_table(self)
            player.reset()
        if self.is_full():
            self.deal_poker()
            self.room.on_table_changed(self)
            logger.info('TABLE[%s] GAME BEGIN[%s]', self.uid, self.players[0].uid)

    def ai_join(self, nth=1):
        size = self.size()
        if size == 0 or size == 3:
            return

        if size == 2 and nth == 1:
            IOLoop.current().call_later(1, self.ai_join, nth=2)

        p1 = AiPlayer(11, 'IDIOT-I', self.players[0])
        p1.to_server([Pt.REQ_JOIN_TABLE, self.uid])

        if size == 1:
            p2 = AiPlayer(12, 'IDIOT-II', self.players[0])
            p2.to_server([Pt.REQ_JOIN_TABLE, self.uid])

    def sync_table(self):
        ps = []
        for p in self.players:
            if p:
                ps.append((p.uid, p.name))
            else:
                ps.append((-1, ''))
        response = [Pt.RSP_JOIN_TABLE, self.uid, ps]
        for player in self.players:
            if player:
                player.send(response)

    def deal_poker(self):
        # if not all(p and p.ready for p in self.players):
        #     return

        self.state = Table.PLAYING
        self.pokers = [i for i in range(54)]
        random.shuffle(self.pokers)
        for i in range(51):
            self.players[i % 3].hand_pokers.append(self.pokers.pop())

        self.whose_turn = random.randint(0, 2)
        for p in self.players:
            p.hand_pokers.sort()

            response = [Pt.RSP_DEAL_POKER, self.turn_player.uid, p.hand_pokers]
            p.send(response)

    def call_score_end(self):
        self.call_score = self.max_call_score
        self.whose_turn = self.max_call_score_turn
        self.turn_player.role = 2
        self.turn_player.hand_pokers += self.pokers
        response = [Pt.RSP_SHOW_POKER, self.turn_player.uid, self.pokers]
        for p in self.players:
            p.send(response)
        logger.info('Player[%d] IS LANDLORD[%s]', self.turn_player.uid, str(self.pokers))
        # assign models for AI
        this_agent_idx = 0
        next_agent_idx = 0
        if self.turn_player.uid == 11:
            this_agent_idx = self.whose_turn
            while self.players[next_agent_idx].uid != 12:
                next_agent_idx += 1
            this_agent_name = 'agent1'
            next_agent_name = 'agent2'
        elif self.turn_player == 12:
            this_agent_idx = self.whose_turn
            while self.players[next_agent_idx].uid != 11:
                next_agent_idx += 1
            this_agent_name = 'agent1'
            next_agent_name = 'agent3'
        else:
            while self.players[next_agent_idx].uid != 12:
                next_agent_idx += 1
            while self.players[this_agent_idx].uid != 11:
                this_agent_idx += 1
            this_agent_name = 'agent2'
            next_agent_name = 'agent3'

        agent_names = ['agent%d' % i for i in range(1, 4)]
        self.players[this_agent_idx].predictor = Predictor(OfflinePredictor(PredictConfig(
            model=Model(agent_names, STATE_SHAPE, METHOD, NUM_ACTIONS, GAMMA),
            session_init=SaverRestore(MODEL_PATH),
            input_names=[this_agent_name + '/state', this_agent_name + '_comb_mask', this_agent_name + '/fine_mask'],
            output_names=[this_agent_name + '/Qvalue']
        )))
        self.players[next_agent_idx].predictor = Predictor(OfflinePredictor(PredictConfig(
            model=Model(agent_names, STATE_SHAPE, METHOD, NUM_ACTIONS, GAMMA),
            session_init=SaverRestore(MODEL_PATH),
            input_names=[next_agent_name + '/state', next_agent_name + '_comb_mask', next_agent_name + '/fine_mask'],
            output_names=[next_agent_name + '/Qvalue']
        )))

    def go_next_turn(self):
        if self.turn_player.become_controller:
            self.controller = self.whose_turn
        self.whose_turn += 1
        if self.whose_turn == 3:
            self.whose_turn = 0

    def get_last_two_cards(self):
        return [self.out_cards[(self.whose_turn + 2) % 3],
                self.out_cards[(self.whose_turn + 1) % 3]]

    @property
    def turn_player(self):
        return self.players[self.whose_turn]

    def handle_chat(self, player, msg):
        response = [Pt.RSP_CHAT, player.uid, msg]
        for p in self.players:
            p.send(response)

    def on_join(self, player):
        if self.is_full():
            logger.error('Player[%d] JOIN Table[%d] FULL', player.uid, self.uid)
        for i, p in enumerate(self.players):
            if not p:
                player.seat = i
                self.players[i] = player
                self.history[i] = []
                break
        self.sync_table()

    def on_leave(self, player):
        for i, p in enumerate(self.players):
            if p == player:
                self.players[i] = None
                self.history[i] = None
                break

    def on_game_over(self, winner):
        # if winner.hand_pokers:
        #     return
        coin = self.room.entrance_fee * self.call_score * self.multiple
        for p in self.players:
            response = [Pt.RSP_GAME_OVER, winner.uid, coin if p != winner else coin * 2 - 100]
            for pp in self.players:
                if pp != p:
                    response.append([pp.uid, *pp.hand_pokers])
            p.send(response)
        # TODO deduct coin from database
        # TODO store poker round to database
        logger.info('Table[%d] GameOver[%d]', self.uid, self.uid)

    def remove(self, player):
        for i, p in enumerate(self.players):
            if p and p.uid == player.uid:
                self.players[i] = None
                self.history[i] = None
        else:
            logger.error('Player[%d] NOT IN Table[%d]', player.uid, self.uid)

        if all(p is None for p in self.players):
            self.state = 3
            logger.error('Table[%d] close', self.uid)
            return True
        return False

    def is_full(self):
        return self.size() == 3

    def is_empty(self):
        return self.size() == 0

    def size(self):
        return sum([p is not None for p in self.players])

    def __str__(self):
        return '[{}: {}]'.format(self.uid, self.players)

    def all_called(self):
        for p in self.players:
            if not p.is_called:
                return False
        return True


