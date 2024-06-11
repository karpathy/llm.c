#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <nccl.h>

#define MAX_RETRIES 5
#define MAX_PROCS 1024

static int is_server = 0;
static int n_proc = 0;
static int proc_id = 0;
static char server_ip[INET_ADDRSTRLEN];
static int server_port = 0;
static float values[MAX_PROCS];
static ncclUniqueId nccl_id;
static int client_count = 0;
static int client_socket = -1;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

typedef enum {
    REDUCE,
    BROADCAST
} OperationType;

void *handle_client(void *arg) {
    int client_socket = (intptr_t)arg;

    while (1) {
        int proc_id_client;
        if (read(client_socket, &proc_id_client, sizeof(int)) <= 0) {
            perror("Read proc_id_client failed");
            close(client_socket);
            return NULL;
        }

        OperationType op_type;
        if (read(client_socket, &op_type, sizeof(OperationType)) <= 0) {
            perror("Read op_type failed");
            close(client_socket);
            return NULL;
        }

        printf("\nClient process id: %d connected to server, Operation type: %d requested\n", proc_id_client, op_type);

        pthread_mutex_lock(&mutex);

        if (op_type == REDUCE) {
            float val;
            if (read(client_socket, &val, sizeof(float)) <= 0) {
                perror("Read val failed");
                close(client_socket);
                pthread_mutex_unlock(&mutex);
                return NULL;
            }
            values[client_count] = val;
            client_count++;

            if (client_count == n_proc) {
                float sum = 0.0;
                for (int i = 0; i < n_proc; i++) {
                    sum += values[i];
                }
                values[0] = sum;
                client_count = 0;
                pthread_cond_broadcast(&cond);
            } else {
                pthread_cond_wait(&cond, &mutex);
            }
            if (write(client_socket, &values[0], sizeof(float)) <= 0) {
                perror("Write result failed");
            }
        } else if (op_type == BROADCAST) {
            ncclUniqueId dummy_id;
            if (read(client_socket, &dummy_id, sizeof(ncclUniqueId)) <= 0) {
                perror("Read dummy_id failed");
                close(client_socket);
                pthread_mutex_unlock(&mutex);
                return NULL;
            }
            client_count++;

            if (client_count == n_proc) {
                client_count = 0;
                pthread_cond_broadcast(&cond);
            } else {
                pthread_cond_wait(&cond, &mutex);
            }
            if (write(client_socket, &nccl_id, sizeof(ncclUniqueId)) <= 0) {
                perror("Write nccl_id failed");
            }
        }

        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

void *server_function(void *arg) {
    int server_fd, client_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) != 0) {
        perror("Setsockopt failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(server_port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, n_proc) < 0) {
        perror("Listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("\nServer started\n");

    while (1) {
        if ((client_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
            perror("Accept failed");
            close(server_fd);
            exit(EXIT_FAILURE);
        }

        pthread_t thread_id;
        if (pthread_create(&thread_id, NULL, handle_client, (void *)(intptr_t)client_socket) != 0) {
            perror("Thread creation failed");
            close(client_socket);
        }
        pthread_detach(thread_id);
    }
    return NULL;
}

void client_init() {
    struct sockaddr_in serv_addr;

    if ((client_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation error");
        exit(EXIT_FAILURE);
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(server_port);

    if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
        fprintf(stderr, "Invalid server address\n");
        close(client_socket);
        exit(EXIT_FAILURE);
    }

    int n_retry = 0;
    while (connect(client_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        if (++n_retry > MAX_RETRIES) {
            fprintf(stderr, "Connection failed to server after %d retries\n", MAX_RETRIES);
            close(client_socket);
            exit(EXIT_FAILURE);
        }
        printf("Connection failed, retrying...\n");
        sleep(1);
    }
}

void client_cleanup() {
    if (client_socket != -1) {
        close(client_socket);
    }
}

void client_function(void *val, size_t data_size, OperationType op_type) {
    if (write(client_socket, &proc_id, sizeof(int)) <= 0) {
        perror("Write proc_id failed");
        return;
    }

    if (write(client_socket, &op_type, sizeof(OperationType)) <= 0) {
        perror("Write op_type failed");
        return;
    }

    if (write(client_socket, val, data_size) <= 0) {
        perror("Write val failed");
        return;
    }

    if (read(client_socket, val, data_size) <= 0) {
        perror("Read result failed");
    }
}

void distributed_init(int n_proc_param, int proc_id_param, const char *server_ip_param, int server_port_param) {
    n_proc = n_proc_param;
    proc_id = proc_id_param;
    strcpy(server_ip, server_ip_param);
    server_port = server_port_param;

    if (n_proc == 1) return;

    if (proc_id == 0) {
        is_server = 1;
        pthread_t server_thread;
        if (pthread_create(&server_thread, NULL, server_function, NULL) != 0) {
            perror("Server thread creation failed");
            exit(EXIT_FAILURE);
        }
    } else {
        is_server = 0;
        client_init();
    }
}

void distributed_reduce(float *val) {
    if (n_proc == 1) return;

    if (is_server) {
        pthread_mutex_lock(&mutex);
        values[client_count] = *val;
        client_count++;
        if (client_count == n_proc) {
            float sum = 0.0;
            for (int i = 0; i < n_proc; i++) {
                sum += values[i];
            }
            values[0] = sum;
            client_count = 0;
            *val = sum;
            pthread_cond_broadcast(&cond);
        } else {
            pthread_cond_wait(&cond, &mutex);
            *val = values[0];
        }
        pthread_mutex_unlock(&mutex);
    } else {
        client_function(val, sizeof(float), REDUCE);
    }
}

void distributed_broadcast(ncclUniqueId *id) {
    if (n_proc == 1) return;

    if (is_server) {
        pthread_mutex_lock(&mutex);
        nccl_id = *id;
        client_count++;
        if (client_count == n_proc) {
            pthread_cond_broadcast(&cond);
        } else {
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);
    } else {
        client_function(id, sizeof(ncclUniqueId), BROADCAST);
    }
}

void distributed_barrier() {
    if (n_proc == 1) return;
    float dummy = 0.0;
    distributed_reduce(&dummy);
}
