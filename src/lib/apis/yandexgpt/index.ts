import { WEBUI_API_BASE_URL } from '$lib/constants';

export const getYandexGPTConfig = async (token: string = '') => {
	let error = null;
	const res = await fetch(`${WEBUI_API_BASE_URL}/yandexgpt/config`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err;
			return null;
		});
	if (error) throw error;
	return res;
};

export const updateYandexGPTConfig = async (token: string = '', config: any) => {
	let error = null;
	const res = await fetch(`${WEBUI_API_BASE_URL}/yandexgpt/config/update`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		},
		body: JSON.stringify(config)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err;
			return null;
		});
	if (error) throw error;
	return res;
};

export const getYandexGPTModels = async (token: string = '') => {
	let error = null;
	const res = await fetch(`${WEBUI_API_BASE_URL}/yandexgpt/models`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err;
			return [];
		});
	if (error) throw error;
	return res;
};
